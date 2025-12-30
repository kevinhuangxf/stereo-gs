"""
Download script for Stereo-GS model from Hugging Face Hub.

Usage:
    python download.py                  # Download checkpoint only
    python download.py --with-data      # Download checkpoint + sample data (and extract)
"""

import os
import zipfile
import argparse
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "kevinhuangxf/stereo-gs"

def main():
    parser = argparse.ArgumentParser(description="Download Stereo-GS from Hugging Face Hub")
    parser.add_argument("--with-data", action="store_true", help="Also download sample data")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, "ckpts")
    
    # Create directories
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Download checkpoint
    print(f"[INFO] Downloading checkpoint from {REPO_ID}...")
    try:
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="model.safetensors",
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False
        )
        print(f"[SUCCESS] Checkpoint downloaded to: {ckpt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download checkpoint: {e}")
        return
    
    # Optionally download sample data
    if args.with_data:
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"[INFO] Downloading sample data...")
        try:
            # Download entire data folder
            snapshot_download(
                repo_id=REPO_ID,
                allow_patterns=["data/*"],
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"[SUCCESS] Sample data downloaded to: {data_dir}")
            
            # Extract zip file if exists
            zip_path = os.path.join(data_dir, "gso_test_data.zip")
            if os.path.exists(zip_path):
                print(f"[INFO] Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"[SUCCESS] Extracted to: {data_dir}/gso_data_lgm_eval_fix/")
                
                # Remove zip after extraction to save space
                os.remove(zip_path)
                print(f"[INFO] Removed zip file to save space")
            
        except Exception as e:
            print(f"[WARN] Failed to download sample data: {e}")
    
    print(f"\n[DONE] Model ready!")
    print(f"\nCheckpoint: {ckpt_dir}/model.safetensors")
    if args.with_data:
        print(f"Test data:  {output_dir}/data/gso_data_lgm_eval_fix/")
        print(f"JSON file:  {output_dir}/data/gso_test_4v.json")
    print(f"\nRun evaluation with:")
    print(f"  python test.py stereogs \\")
    print(f"      --resume {ckpt_dir}/model.safetensors \\")
    print(f"      --workspace ./workspace_eval \\")
    print(f"      --eval-views-path-json {output_dir}/data/gso_test_4v.json")

if __name__ == "__main__":
    main()
