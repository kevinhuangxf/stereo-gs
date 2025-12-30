"""
Evaluation script for StereoGS model.
Simplified version of eval.py specifically for StereoGS evaluation.
"""

import os
import tyro
import numpy as np

import torch
from core.options import AllConfigs
from core.stereogs import StereoGS
from safetensors.torch import load_file

import kiui

# Device setup
device = 'cuda'


def main():
    opt = tyro.cli(AllConfigs)
    
    # Create output workspace directory
    os.makedirs(opt.workspace, exist_ok=True)
    
    print("=" * 80)
    print(f"StereoGS Evaluation")
    print(f"Workspace: {opt.workspace}")
    print(f"Resume from: {opt.resume}")
    print(f"Data mode: {opt.data_mode}")
    print("=" * 80)
    
    # Initialize StereoGS model
    model = StereoGS(opt)
    
    # Load checkpoint
    if opt.resume is not None:
        print(f"Loading checkpoint from: {opt.resume}")
        
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # Remap keys for backward compatibility
        # Handle different checkpoint formats:
        # - mast3r_model.* -> stereo_encoder.*
        # - encoder.* -> stereo_encoder.* (for standalone encoder checkpoints)
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
                    print(f'[WARN] Mismatching shape for param {k}: '
                          f'ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
                    mismatched += 1
            else:
                print(f'[WARN] Unexpected param {k}: {v.shape}')
                unexpected += 1
        
        print(f"Checkpoint loaded: {matched} matched, {mismatched} mismatched, {unexpected} unexpected")
    else:
        print("[WARN] No checkpoint provided, using random initialization")
    
    # Load dataset
    if opt.data_mode == 's3':
        from core.provider_objaverse_dust3r import ObjaverseDataset as Dataset
    elif opt.data_mode == 'dust3r':
        from core.provider_objaverse_dust3r import ObjaverseDataset as Dataset
    elif opt.data_mode == 'shapesplat':
        from core.provider_objaverse_dust3r import ShapeSplatDataset as Dataset
    else:
        raise NotImplementedError(f"Data mode {opt.data_mode} not supported")
    
    # Create test dataset (evaluation only, no training dataset)
    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of batches: {len(test_dataloader)}")
    
    # Move model to GPU
    model = model.to(device)
    
    # Evaluation loop
    print("Starting evaluation...")
    
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    num_samples = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # Move data to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
                elif isinstance(data[key], list):
                    for j in range(len(data[key])):
                        if isinstance(data[key][j], dict):
                            for k in data[key][j]:
                                if isinstance(data[key][j][k], torch.Tensor):
                                    data[key][j][k] = data[key][j][k].to(device)
            
            # Forward pass
            out = model(data)
            
            # Accumulate metrics
            psnr = out['psnr']
            total_psnr += psnr.detach()
            ssim = out['ssim']
            total_ssim += ssim
            
            if 'lpips' in out:
                lpips = out['lpips']
                total_lpips += lpips.detach()
            
            num_samples += 1
            
            # Save visualization images
            gt_images = data['images_output'].detach().cpu().numpy()  # [B, V, 3, H, W]
            gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(
                -1, gt_images.shape[1] * gt_images.shape[3], 3
            )
            
            pred_images = out['images_pred'].detach().cpu().numpy()  # [B, V, 3, H, W]
            pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(
                -1, pred_images.shape[1] * pred_images.shape[3], 3
            )
            
            # Concatenate GT and prediction vertically
            images_combined = np.concatenate((gt_images, pred_images), axis=0)
            kiui.write_image(f'{opt.workspace}/eval_{i:04d}.jpg', images_combined)
            
            # Save predicted alpha masks
            pred_alphas = out['alphas_pred'].detach().cpu().numpy()  # [B, V, 1, H, W]
            pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(
                -1, pred_alphas.shape[1] * pred_alphas.shape[3], 1
            )
            kiui.write_image(f'{opt.workspace}/eval_alpha_{i:04d}.jpg', pred_alphas)
            
            # Progress update
            if (i + 1) % 10 == 0 or i == 0:
                avg_psnr = total_psnr / num_samples
                avg_ssim = total_ssim / num_samples
                print(f"[{i+1}/{len(test_dataloader)}] "
                      f"PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    # Final metrics
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples if total_lpips > 0 else 0
    
    print("=" * 80)
    print("Evaluation Results:")
    print(f"  Average PSNR:  {avg_psnr:.2f} dB")
    print(f"  Average SSIM:  {avg_ssim:.4f}")
    if avg_lpips > 0:
        print(f"  Average LPIPS: {avg_lpips:.4f}")
    print("=" * 80)
    
    # Save results to file
    with open(f'{opt.workspace}/eval_results.txt', 'w') as f:
        f.write(f"StereoGS Evaluation Results\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Checkpoint: {opt.resume}\n")
        f.write(f"Data mode: {opt.data_mode}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"\n")
        f.write(f"Average PSNR:  {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
        if avg_lpips > 0:
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write(f"=" * 80 + "\n")
    
    print(f"Results saved to {opt.workspace}/eval_results.txt")
    print(f"Visualization images saved to {opt.workspace}/")


if __name__ == '__main__':
    main()
