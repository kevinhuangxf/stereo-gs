"""
Inference script for StereoGS model with MVDream input.
Generates 3D Gaussian Splats from a single image using MVDream for multi-view generation.
"""

import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import rembg

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs
from core.stereogs import StereoGS
from mvdream.pipeline_mvdream import MVDreamPipeline

from dust3r.utils.image import load_images
from dust3r.inference import inference
from core.utils import get_rays

import time

minmax_norm = lambda x: (x - x.min()) / (x.max() - x.min())

# Parse options
opt = tyro.cli(AllConfigs)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize StereoGS model
print(f"[INFO] Initializing StereoGS model...")
model = StereoGS(opt)

# Load checkpoint
if opt.resume is not None:
    print(f"[INFO] Loading checkpoint from: {opt.resume}")
    
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    
    # Remap keys for backward compatibility
    remapped_ckpt = {}
    remap_count = 0
    for k, v in ckpt.items():
        new_k = k
        if k.startswith('mast3r_model.'):
            new_k = k.replace('mast3r_model.', 'stereo_encoder.')
            remap_count += 1
        elif k.startswith('encoder.'):
            new_k = k.replace('encoder.', 'stereo_encoder.')
            remap_count += 1
        remapped_ckpt[new_k] = v
    if remap_count > 0:
        print(f"[INFO] Remapped {remap_count} checkpoint keys to stereo_encoder.*")
    ckpt = remapped_ckpt
    
    # Tolerant load (only load matching shapes)
    state_dict = model.state_dict()
    matched = 0
    mismatched = 0
    unexpected = 0
    
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
                matched += 1
            else:
                mismatched += 1
        else:
            unexpected += 1
    
    print(f"[INFO] Checkpoint loaded: {matched} matched, {mismatched} mismatched, {unexpected} unexpected")
else:
    print(f'[WARN] Model randomly initialized, are you sure?')

model = model.to(device)
model.eval()

# Projection matrix for rendering
tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# Load MVDream pipeline
print("[INFO] Loading MVDream pipeline...")
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipe.to(device)

# Load rembg for background removal
bg_remover = rembg.new_session()


def process(opt, path):
    """Process a single image and generate 3D Gaussian Splats."""
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)
    
    # Load and preprocess input image
    input_image = kiui.read_image(path, mode='uint8')
    
    # Background removal
    carved_image = rembg.remove(input_image, session=bg_remover)  # [H, W, 4]
    mask = carved_image[..., -1] > 0
    
    # Recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # RGBA to RGB with alpha blending
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4]
    
    # Generate multi-view images using MVDream
    print("[INFO] Generating multi-view images with MVDream...")
    tik = time.time()
    mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)  # [4, 256, 256, 3]
    tok = time.time()
    print(f'[INFO] MVDream took {tok - tik:.2f}s')
    
    # Save intermediate multi-view images
    mv_dir = os.path.join(opt.workspace, 'mv_images')
    os.makedirs(mv_dir, exist_ok=True)
    for i in range(4):
        imageio.imwrite(os.path.join(mv_dir, f'{name}_mv_{i}.png'), (mv_image[i] * 255).astype(np.uint8))
    
    with torch.no_grad():
        # Load images for StereoGS
        mv_paths = [os.path.join(mv_dir, f'{name}_mv_{i}.png') for i in range(4)]
        images = load_images(mv_paths, size=256, square_ok=True, verbose=False)
        
        # Prepare image pairs for stereo encoder
        for i in range(4):
            if i % 2 == 0 and i != 0:
                images[0]['img'] = torch.cat((images[0]['img'], images[i]['img']), dim=0).unsqueeze(0)
            elif i % 2 == 1 and i != 1:
                images[1]['img'] = torch.cat((images[1]['img'], images[i]['img']), dim=0).unsqueeze(0)
        
        # Prepare camera poses
        cam_poses = np.stack([
            orbit_camera(0, 0, radius=opt.cam_radius),
            orbit_camera(0, 90, radius=opt.cam_radius),
            orbit_camera(0, 180, radius=opt.cam_radius),
            orbit_camera(0, 270, radius=opt.cam_radius),
        ], axis=0)
        cam_poses = torch.from_numpy(cam_poses)
        
        # Prepare rays
        rays = []
        for i in [0, 2, 1, 3]:
            rays_o, rays_d = get_rays(cam_poses[i], 256, 256, opt.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays.append(rays_plucker)
        mast3r_rays = torch.stack(rays, dim=0).permute(0, 3, 1, 2).contiguous()  # [V, 6, h, w]
        
        # Reshape images
        B, V, C, H, W = images[0]['img'].shape
        images[0]['img'] = images[0]['img'].view(B * V, C, H, W)
        images[1]['img'] = images[1]['img'].view(B * V, C, H, W)
        images = images[:2]
        cat_images = torch.cat([
            images[0]['img'].view(B, V, C, H, W),
            images[1]['img'].view(B, V, C, H, W)
        ], dim=1)
        
        # Forward pass through StereoGS
        print("[INFO] Running StereoGS inference...")
        tik = time.time()
        
        # Use dust3r inference to get stereo features
        device_id = torch.cuda.current_device()
        output = inference(
            [tuple(images)],
            model.stereo_encoder.to(f"cuda:{device_id}"),
            f"cuda:{device_id}",
            batch_size=V,
            verbose=False
        )
        
        # Extract predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        if model.use_unet_refinement:
            # UNet mode: use desc features
            N, H_feat, W_feat, C_feat = pred1['desc'].shape
            pred1['desc'] = pred1['desc'].view(B, V, H_feat, W_feat, C_feat)
            pred2['desc'] = pred2['desc'].view(B, V, H_feat, W_feat, C_feat)
            
            # Concat features and rays
            feat1 = pred1['desc'].permute(0, 1, 4, 2, 3)
            feat2 = pred2['desc'].permute(0, 1, 4, 2, 3)
            feat_mast3r = torch.cat([feat1, feat2], dim=1)
            
            cat_input = torch.cat([cat_images.cuda(), feat_mast3r.cuda(), mast3r_rays.unsqueeze(0).cuda()], dim=2)
            gaussians = model.forward_unet(cat_input)
        else:
            # Direct mode: use gaussians from pred
            gaussians = pred1.get('gaussians', pred1.get('pts3d', None))
            if gaussians is None:
                raise ValueError("Direct Gaussian mode requires 'gaussians' in prediction output")
        
        tok = time.time()
        print(f'[INFO] StereoGS inference took {tok - tik:.2f}s')
        
        # Save Gaussian PLY
        ply_path = os.path.join(opt.workspace, name + '.ply')
        model.gs.save_ply(gaussians, ply_path)
        print(f'[INFO] Saved Gaussian PLY to {ply_path}')
        
        # Render 360 video
        print("[INFO] Rendering 360 video...")
        images_rendered = []
        depths = []
        elevation = -10
        render_radius = 1.85
        
        if getattr(opt, 'fancy_video', False):
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                cam_poses_render = torch.from_numpy(
                    orbit_camera(elevation, azi, radius=render_radius, opengl=True)
                ).unsqueeze(0).to(device)
                cam_poses_render[:, :3, 1:3] *= -1
                
                cam_view = torch.inverse(cam_poses_render).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                cam_pos = -cam_poses_render[:, :3, 3]
                
                scale = min(azi / 360, 1)
                image = model.gs.render(
                    gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0), scale_modifier=scale
                )['image']
                images_rendered.append(
                    (image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                )
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                cam_poses_render = torch.from_numpy(
                    orbit_camera(elevation, azi, radius=render_radius, opengl=True)
                ).unsqueeze(0).to(device)
                cam_poses_render[:, :3, 1:3] *= -1
                
                cam_view = torch.inverse(cam_poses_render).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                cam_pos = -cam_poses_render[:, :3, 3]
                
                results = model.gs.render(
                    gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0), scale_modifier=1
                )
                depth = minmax_norm(results['depth'])
                alpha = results['alpha']
                depth[alpha < 0.5] = 1
                image = results['image']
                depths.append(
                    (depth.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                )
                images_rendered.append(
                    (image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                )
        
        # Save videos
        images_rendered = np.concatenate(images_rendered, axis=0)
        video_path = os.path.join(opt.workspace, name + '.mp4')
        imageio.mimwrite(video_path, images_rendered, fps=30)
        print(f'[INFO] Saved video to {video_path}')
        
        if depths:
            depths = np.concatenate(depths, axis=0)
            depth_video_path = os.path.join(opt.workspace, name + '_depth.mp4')
            imageio.mimwrite(depth_video_path, depths, fps=30)
            print(f'[INFO] Saved depth video to {depth_video_path}')


# Main execution
if __name__ == '__main__':
    assert opt.test_path is not None, "Please provide --test-path to the input image(s)"
    
    if os.path.isdir(opt.test_path):
        file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    else:
        file_paths = [opt.test_path]
    
    print(f"[INFO] Processing {len(file_paths)} file(s)...")
    for path in file_paths:
        process(opt, path)
    
    print("[INFO] Done!")

