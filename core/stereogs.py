"""
StereoGS: Unified Stereo Gaussian Splatting Model

Combines LGM and MASt3RGS approaches with configuration-based switching:
- LGM mode: Frozen MASt3R for features → UNet → Gaussians
- MASt3RGS mode: Trainable MASt3R heads → Direct Gaussians (optional MV attention)
- Hybrid modes: Combinations with optional UNet refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer
from kiui.lpips import LPIPS

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Initialize the SSIM metric
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

minmax_norm = lambda x: (x - x.min()) / (x.max() - x.min())


class StereoGS(nn.Module):

    def __init__(self, opt: Options):
        super().__init__()
        
        self.opt = opt
        
        # Configuration flag: use_unet_refinement
        # - True (default): Pretrained MASt3R → desc features → UNet → Gaussians
        # - False: MASt3R with MV Gaussian head → Gaussians directly
        self.use_unet_refinement = getattr(opt, 'use_unet_refinement', True)
        
        # Initialize stereo encoder based on mode
        self._init_stereo_encoder()
        
        # Initialize UNet if refinement is needed
        if self.use_unet_refinement:
            self._init_unet()
        
        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)
        
        # Activation functions for Gaussian parameters
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5
        
        # LPIPS loss for perceptual quality
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
    
    def _init_stereo_encoder(self):
        """
        Initialize stereo encoder based on use_unet_refinement flag.
        
        - use_unet_refinement=True: Pretrained MASt3R with original head (outputs desc)
        - use_unet_refinement=False: MASt3R with MV Gaussian head (outputs Gaussians)
        """
        # device = 'cuda'
        # model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        # self.stereo_encoder = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        # self.stereo_encoder.eval()
        
        if self.use_unet_refinement:
            # UNet mode: MASt3R with catmlp+dpt head → desc features → UNet → Gaussians
            print("[StereoGS] Initializing MASt3R for feature extraction (UNet mode)")
            
            self.stereo_encoder = AsymmetricMASt3R(
                pos_embed='RoPE100',
                patch_embed_cls='PatchEmbedDust3R',
                img_size=(512, 512),
                # head_type='catmlp+dpt',  # Original MASt3R head that outputs 'desc' only
                # output_mode='pts3d+desc24',
                head_type='gaussian_head',  # Unified head that outputs 'desc' + 'gaussians'
                output_mode='pts3d+gaussian+desc24',
                depth_mode=('exp', -float('inf'), float('inf')),
                conf_mode=('exp', 1, float('inf')),
                enc_embed_dim=1024,
                enc_depth=24,
                enc_num_heads=16,
                dec_embed_dim=768,
                dec_depth=12,
                dec_num_heads=12,
                two_confs=True,
                use_offsets=getattr(self.opt, 'use_offsets', True),
                sh_degree=getattr(self.opt, 'sh_degree', 1)
            )
            
            # Note: Checkpoint loading is handled externally in eval_stereogs.py
            # Initialize with random weights - actual checkpoint loaded later
            
            self.stereo_encoder.eval()
            
            # Freeze all parameters
            for param in self.stereo_encoder.parameters():
                param.requires_grad = False
            
            print("[StereoGS] MASt3R fully frozen for feature extraction")
            
        else:
            # Direct Gaussian mode: MASt3R with MV Gaussian head
            print("[StereoGS] Initializing MASt3R with MV Gaussian head (direct Gaussian mode)")
            
            self.stereo_encoder = AsymmetricMASt3R(
                pos_embed='RoPE100',
                patch_embed_cls='PatchEmbedDust3R',
                img_size=(512, 512),
                head_type='mv_head',  # Creates MVGaussianHead with mv_attn
                output_mode='pts3d+gaussian+desc24',
                depth_mode=('exp', -float('inf'), float('inf')),
                conf_mode=('exp', 1, float('inf')),
                enc_embed_dim=1024,
                enc_depth=24,
                enc_num_heads=16,
                dec_embed_dim=768,
                dec_depth=12,
                dec_num_heads=12,
                two_confs=True,
                use_offsets=getattr(self.opt, 'use_offsets', True),
                sh_degree=getattr(self.opt, 'sh_degree', 1)
            )
            
            # Note: Checkpoint loading is handled externally in eval_stereogs.py
            # Initialize with random weights - actual checkpoint loaded later
            
            # Freeze encoder and decoder, keep MV head trainable
            for param in self.stereo_encoder.patch_embed.parameters():
                param.requires_grad = False
            for param in self.stereo_encoder.enc_blocks.parameters():
                param.requires_grad = False
            for param in self.stereo_encoder.dec_blocks.parameters():
                param.requires_grad = False
            
            print("[StereoGS] Encoder/Decoder frozen, MV Gaussian head trainable")
    
    def _init_unet(self):
        """Initialize UNet for Gaussian parameter prediction (only when use_unet_refinement=True)."""
        # Determine input channels: RGB (3) + MASt3R desc features (24) + rays (6) = 33
        if self.opt.data_mode == 'dust3r':
            in_channels = 33  # RGB (3) + desc (24) + rays (6)
        else:
            in_channels = 9   # RGB (3) + rays (6)
        
        print(f"[StereoGS] Initializing UNet with {in_channels} input channels")
        
        self.unet = UNet(
            in_channels, 14,  # Output: pos(3) + opacity(1) + scale(3) + rotation(4) + rgb(3)
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=self.opt.num_input_views,
        )
        
        # Final convolution layer
        self.conv = nn.Conv2d(14, 14, kernel_size=1)
    
    def state_dict(self, **kwargs):
        """Remove LPIPS loss from state dict to avoid saving it."""
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with automatic key remapping for backward compatibility.
        
        Handles checkpoint key differences:
        - Old checkpoints use 'mast3r_model.' prefix
        - New unified model uses 'stereo_encoder.' prefix
        """
        # Remap keys for backward compatibility
        remapped_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            # Handle old LGM checkpoints: mast3r_model -> stereo_encoder
            if k.startswith('mast3r_model.'):
                new_k = k.replace('mast3r_model.', 'stereo_encoder.')
                print(f"[StereoGS] Remapping checkpoint key: {k} -> {new_k}")
            remapped_state_dict[new_k] = v
        
        return super().load_state_dict(remapped_state_dict, strict=strict)
    
    def forward_unet(self, cat_input):
        """
        UNet refinement: Predict Gaussian parameters from concatenated features.
        
        This is step 2 when use_unet_refinement=True.
        Takes feat_mast3r from forward_stereo_model and predicts Gaussians.
        
        Args:
            cat_input: [B, V, C, H, W] - concatenated images + MASt3R features + rays
            
        Returns:
            gaussians: [B, N, 14] - Gaussian parameters
        """
        B, V, C, H, W = cat_input.shape
        cat_input = cat_input.view(B * V, C, H, W)
        
        # UNet prediction
        x = self.unet(cat_input)  # [B*V, 14, h, w]
        x = self.conv(x)  # [B*V, 14, h, w]
        
        # Reshape to [B, V, 14, h, w]
        x = x.reshape(B, V, 14, self.opt.splat_size, self.opt.splat_size)
        
        # Flatten spatial dimensions: [B, N, 14] where N = V * h * w
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        # Apply activation functions
        pos = self.pos_act(x[..., 0:3])
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:14])
        
        # Concatenate all parameters
        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)
        
        return gaussians
    
    def forward_stereo_model(self, view1, view2, data, cat_images):
        """
        Unified forward pass through stereo model (MASt3R).
        
        Both paths share the same encoder-decoder, only the head differs:
        - use_unet_refinement=True: Original MASt3R head → desc features for UNet
        - use_unet_refinement=False: MV Gaussian head → Gaussians directly
        
        Args:
            view1, view2: Dictionaries with 'img' tensors
            data: Full data dictionary
            cat_images: [B, V, C, H, W] - concatenated input images
            
        Returns:
            If use_unet_refinement=True:
                feat_mast3r: [B, V, C, H, W] - MASt3R features for UNet
            If use_unet_refinement=False:
                gaussians: [B, N, 14] - Gaussian parameters
        """
        B, V, C, H, W = cat_images.shape
        
        # ============================================================
        # Step 1: Common encoder-decoder (same for both paths)
        # ============================================================
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.stereo_encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.stereo_encoder._decoder(feat1, pos1, feat2, pos2)
        
        # ============================================================
        # Step 2: Branch based on use_unet_refinement
        # ============================================================
        if self.use_unet_refinement:
            # Original MASt3R head → desc features for UNet
            pred1 = self.stereo_encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
            pred2 = self.stereo_encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)
            
            # Extract desc features
            # pred['desc'] shape: [N, H, W, C] where N = B * V // 2
            N, feat_H, feat_W, feat_C = pred1['desc'].shape
            desc1 = pred1['desc'].view(B, V // 2, feat_H, feat_W, feat_C)
            desc2 = pred2['desc'].view(B, V // 2, feat_H, feat_W, feat_C)
            
            # Permute to [B, V//2, C, H, W] and concatenate
            feat1 = desc1.permute(0, 1, 4, 2, 3)
            feat2 = desc2.permute(0, 1, 4, 2, 3)
            feat_mast3r = torch.cat([feat1, feat2], dim=1)  # [B, V, C, H, W]
            
            return feat_mast3r
        
        else:
            # MV Gaussian head → Gaussians directly
            H_out, W_out = self.opt.output_size, self.opt.output_size
            
            # Interleave dec1 and dec2 for proper batch ordering
            dec = []
            for d1, d2 in zip(dec1, dec2):
                d1_reshaped = d1.reshape(-1, 2, *d1.shape[1:])
                d2_reshaped = d2.reshape(-1, 2, *d2.shape[1:])
                d_interleaved = torch.stack([
                    d1_reshaped[:, 0], d2_reshaped[:, 0],
                    d1_reshaped[:, 1], d2_reshaped[:, 1]
                ], dim=1)
                d_interleaved = d_interleaved.reshape(-1, *d1.shape[1:])
                dec.append(d_interleaved)
            
            # Get predictions from MV head
            pred = self.stereo_encoder.mv_head(dec, (H_out, W_out, V))
            
            # Extract Gaussian parameters
            pos = pred[:, :3, ...].reshape(B * V, -1, 3)
            opacity = pred[:, 3:4, ...].reshape(B * V, -1, 1)
            scale = pred[:, 4:7, ...].reshape(B * V, -1, 3)
            rotation = pred[:, 7:11, ...].reshape(B * V, -1, 4)
            rgbs = pred[:, 11:14, ...].reshape(B * V, -1, 3)
            
            # Apply activation functions
            pos = self.pos_act(pos)
            opacity = self.opacity_act(opacity)
            scale = self.scale_act(scale)
            rotation = self.rot_act(rotation)
            rgbs = self.pos_act(rgbs)  # Note: clamp to [-1, 1]
            
            # Concatenate and reshape
            gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)
            gaussians = gaussians.view(B, -1, 14)
            
            return gaussians
    
    def forward(self, data, step_ratio=1):
        results = {}
        loss = 0
        
        # Get batch dimensions
        B, V, C, H, W = data['images'][0]['img'].shape
        
        # Reshape for MASt3R inference
        data['images'][0]['img'] = data['images'][0]['img'].view(B * V, C, H, W)
        data['images'][1]['img'] = data['images'][1]['img'].view(B * V, C, H, W)
        data['images'][0]['true_shape'] = data['images'][0]['true_shape'][0]
        data['images'][1]['true_shape'] = data['images'][1]['true_shape'][0]
        
        images = data['images'][:2]
        cat_images = torch.cat([
            images[0]['img'].view(B, V, C, H, W),
            images[1]['img'].view(B, V, C, H, W)
        ], dim=1)
        
        # Step 1: Forward through stereo model
        # Returns either gaussians (if use_unet_refinement=False) or feat_mast3r (if use_unet_refinement=True)
        stereo_output = self.forward_stereo_model(images[0], images[1], data, cat_images)
        
        # Step 2: Check if we need UNet refinement
        if self.use_unet_refinement:
            # stereo_output is feat_mast3r features
            feat_mast3r = stereo_output
            
            # Concatenate: images + MASt3R features + rays
            cat_input = torch.cat([cat_images, feat_mast3r.cuda(), data['rays']], dim=2)
            
            # Predict Gaussians with UNet
            gaussians = self.forward_unet(cat_input)
        else:
            # stereo_output is already gaussians from MASt3RGS heads
            gaussians = stereo_output
        
        results['gaussians'] = gaussians
        
        # Render with Gaussian splatting
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        results = self.gs.render(
            gaussians,
            data['cam_view'],
            data['cam_view_proj'],
            data['cam_pos'],
            bg_color=bg_color
        )
        
        pred_images = results['image']
        pred_alphas = results['alpha']
        pred_depths = results['depth']
        
        pred_depths = minmax_norm(pred_depths)
        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths
        
        # Ground truth
        gt_images = data['images_output']
        gt_masks = data['masks_output']
        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        
        # Compute losses
        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse
        
        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(
                    gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                    (256, 256),
                    mode='bilinear',
                    align_corners=False
                ),
                F.interpolate(
                    pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                    (256, 256),
                    mode='bilinear',
                    align_corners=False
                ),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
        
        results['loss'] = loss
        
        # Compute metrics
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            ssim_score = ssim(
                pred_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu(),
                gt_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu()
            )
            
            if self.opt.lambda_lpips > 0:
                results['lpips'] = loss_lpips
            results['ssim'] = ssim_score.item()
            results['psnr'] = psnr
            results['gaussians'] = gaussians
        
        return results
