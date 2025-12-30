# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import postprocess
import dust3r.utils.path_to_croco  # noqa: F401
from croco.models.dpt_block import DPTOutputAdapter  # noqa

from core.attention import MemEffAttention

class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mv_attn = False
        if 'mv_attn' in kwargs:
            self.mv_attn = kwargs['mv_attn']
            if self.mv_attn:
                dim0=1024
                dim=768
                num_heads = 16
                qkv_bias = False
                proj_bias = True
                attn_drop = 0.0
                proj_drop = 0.0
                self.norm0 = nn.LayerNorm(dim0)
                self.attn0 = MemEffAttention(dim0, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
                self.norm1 = nn.LayerNorm(dim)
                self.attn1 = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
                self.norm2 = nn.LayerNorm(dim)
                self.attn2 = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
                self.norm3 = nn.LayerNorm(dim)
                self.attn3 = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, camera_params=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        # if len(image_size) == 3:
        if self.mv_attn:
            H, W, V = image_size[0], image_size[1], image_size[2]
        else:
            H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        if self.mv_attn:
            num_input_views = V # hard coded!
            BV, N, C0 = layers[0].shape
            BV, N, C = layers[1].shape
            B = BV // num_input_views
            # layer 0
            layer_0 = layers[0].reshape((B, num_input_views*N, C0))
            layer_0 = layer_0 + self.attn0(self.norm0(layer_0))
            layers[0] = layer_0.reshape((BV, N, C0))
            # layer 1
            layer_1 = layers[1].reshape((B, num_input_views*N, C))
            layer_1 = layer_1 + self.attn1(self.norm1(layer_1))
            layers[1] = layer_1.reshape((BV, N, C))
            # layer 2
            layer_2 = layers[2].reshape((B, num_input_views*N, C))
            layer_2 = layer_2 + self.attn2(self.norm2(layer_2))
            layers[2] = layer_2.reshape((BV, N, C))
            # layer 3
            layer_3 = layers[3].reshape((B, num_input_views*N, C))
            layer_3 = layer_3 + self.attn3(self.norm3(layer_3))
            layers[3] = layer_3.reshape((BV, N, C))
            # layers[0] = layers[0] + self.attn0(self.norm0(layers[0]))
            # layers[1] = layers[1] + self.attn1(self.norm1(layers[1]))
            # layers[2] = layers[2] + self.attn2(self.norm2(layers[2]))
            # layers[3] = layers[3] + self.attn3(self.norm3(layers[3]))

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        if camera_params is not None:
            # Fuse layers using refinement stages
            path_4 = self.scratch.refinenet4(layers[3], camera_params)[:, :, :layers[2].shape[2], :layers[2].shape[3]]
            path_3 = self.scratch.refinenet3(path_4, layers[2], camera_params)
            path_2 = self.scratch.refinenet2(path_3, layers[1], camera_params)
            path_1 = self.scratch.refinenet1(path_2, layers[0], camera_params)
        else:
            # Fuse layers using refinement stages
            path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
            path_3 = self.scratch.refinenet3(path_4, layers[2])
            path_2 = self.scratch.refinenet2(path_3, layers[1])
            path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out #, path_1 # pt3d, depth_features


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        if len(img_info) == 3:
            out = self.dpt(x, image_size=img_info)
        else:
            out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')
