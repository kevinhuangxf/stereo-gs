import os
import re
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import rembg

from kiui.cam import orbit_camera

from core.options import AllConfigs
from core.stereogs import StereoGS

from dust3r.utils.image import load_images
from dust3r.inference import inference
from core.utils import get_rays

import time

minmax_norm = lambda x: (x - x.min()) / (x.max() - x.min())

# Parse options
opt = tyro.cli(AllConfigs)

# Get number of input views from config
NUM_INPUT_VIEWS = opt.num_input_views
print(f"[INFO] Using {NUM_INPUT_VIEWS} input views")

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

# Load rembg for background removal
bg_remover = rembg.new_session()


def process(opt, folder_path):
    """Process a folder containing pre-generated multi-view images."""
    name = os.path.basename(folder_path.rstrip('/'))
    print(f'[INFO] Processing {folder_path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)
    
    # Auto-detect and sort images in folder by natural numeric order
    all_images = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
    total_images = len(all_images)
    print(f'[INFO] Found {total_images} images in {folder_path}')
    
    if total_images < NUM_INPUT_VIEWS:
        raise ValueError(f"Need at least {NUM_INPUT_VIEWS} images, but found only {total_images}")
    
    # Calculate interval for evenly-spaced selection
    interval = max(1, total_images // NUM_INPUT_VIEWS)
    
    # Load and preprocess images
    target_size = 256  # Target resolution
    mv_image = []
    selected_paths = []
    
    for i in range(NUM_INPUT_VIEWS):
        idx = min(i * interval, total_images - 1)
        img_path = os.path.join(folder_path, all_images[idx])
        selected_paths.append(img_path)
        print(f'[INFO] Loading image {i+1}/{NUM_INPUT_VIEWS}: {img_path}')
        
        input_image = imageio.imread(img_path)
        
        # Background removal (always applied, matching mast3r_6v)
        carved_image = rembg.remove(input_image, session=bg_remover)  # [H, W, 4]
        
        # Don't recenter - use carved image directly (matching mast3r_6v)
        image = carved_image
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # RGBA to RGB with alpha blending
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4]
        
        # Resize to target size if needed
        if image.shape[0] != target_size or image.shape[1] != target_size:
            # Use torch for resizing
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            image_tensor = F.interpolate(image_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
            image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, C]
        
        mv_image.append(image)
    
    mv_image = np.stack(mv_image, axis=0)  # [N, 256, 256, 3]
    
    # Save preprocessed images for debugging
    mv_dir = os.path.join(opt.workspace, 'mv_images')
    os.makedirs(mv_dir, exist_ok=True)
    for i in range(NUM_INPUT_VIEWS):
        imageio.imwrite(os.path.join(mv_dir, f'{name}_mv_{i}.png'), (mv_image[i] * 255).astype(np.uint8))
    
    with torch.no_grad():
        # Load images for StereoGS using dust3r's loader
        mv_paths = [os.path.join(mv_dir, f'{name}_mv_{i}.png') for i in range(NUM_INPUT_VIEWS)]
        images = load_images(mv_paths, size=256, square_ok=True, verbose=False)
        
        # Prepare image pairs for stereo encoder
        # Group images: even indices -> images[0], odd indices -> images[1]
        for i in range(NUM_INPUT_VIEWS):
            if i % 2 == 0 and i != 0:
                images[0]['img'] = torch.cat((images[0]['img'], images[i]['img']), dim=0)
            elif i % 2 == 1 and i != 1:
                images[1]['img'] = torch.cat((images[1]['img'], images[i]['img']), dim=0)
        
        images[0]['img'] = images[0]['img'].unsqueeze(0)
        images[1]['img'] = images[1]['img'].unsqueeze(0)
        
        # Prepare camera poses (evenly distributed around 360 degrees at elevation 0)
        cam_poses = np.stack([
            orbit_camera(0, i * (360 // NUM_INPUT_VIEWS), radius=opt.cam_radius) 
            for i in range(NUM_INPUT_VIEWS)
        ], axis=0)
        cam_poses = torch.from_numpy(cam_poses)
        
        # Prepare rays - order to match image grouping (even first, then odd)
        rays = []
        ray_indices = []
        for i in range(NUM_INPUT_VIEWS):
            if i % 2 == 0:
                ray_indices.append(i)
        for i in range(NUM_INPUT_VIEWS):
            if i % 2 != 0:
                ray_indices.append(i)
        
        for i in ray_indices:
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
            N_feat, H_feat, W_feat, C_feat = pred1['desc'].shape
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
        elevation = -10  # Match mast3r_6v
        render_radius = 1.85  # Match mast3r_6v (hardcoded)
        
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
    assert opt.test_path is not None, "Please provide --test-path to the folder containing pre-generated images"
    
    if os.path.isdir(opt.test_path):
        # Check if this folder directly contains images
        all_images = [f for f in os.listdir(opt.test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(all_images) >= NUM_INPUT_VIEWS:
            # Folder with enough images - process directly
            process(opt, opt.test_path)
        else:
            # Multiple subfolders, each with images
            subfolders = sorted([
                os.path.join(opt.test_path, d) 
                for d in os.listdir(opt.test_path) 
                if os.path.isdir(os.path.join(opt.test_path, d))
            ])
            print(f"[INFO] Processing {len(subfolders)} subfolder(s)...")
            for folder in subfolders:
                try:
                    process(opt, folder)
                except Exception as e:
                    print(f"[ERROR] Failed to process {folder}: {e}")
    else:
        raise ValueError(f"--test-path must be a directory containing pre-generated images, got: {opt.test_path}")
    
    print("[INFO] Done!")

