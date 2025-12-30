# Stereo-GS

Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction (ACM MM 2025)

[Paper]() | [Project]()

## Installation

```bash
conda create -n stereogs python=3.10
conda activate stereogs

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Modified Gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

pip install -r requirements.txt
```

## Download Checkpoints

Download the pre-trained checkpoints and place them in the `ckpts/` folder:

```bash
# Download checkpoint (example)
mkdir -p ckpts
# Place your checkpoint files here, e.g.:
# ckpts/model.safetensors
```

## Inference

### Inference with Pre-generated Multi-View Images (e.g., from V3D)

**4 views:**
```bash
python infer.py stereogs \
    --resume ./ckpts/model.safetensors \
    --workspace ./output \
    --test-path ./data_test/v3d_test
```

**6 views:**
```bash
python infer_v3d.py stereogs_6v \
    --resume ./ckpts/model.safetensors \
    --workspace ./output/basketball \
    --test-path ./data_test/basketball
```

The `--test-path` should point to a folder containing images. The script will auto-detect and select evenly-spaced images.

<!-- ### Inference with MVDream

Generate multi-view images from a single RGBA image:

```bash
python infer_mvdream.py stereogs \
    --resume ./ckpts/model.safetensors \
    --workspace ./output \
    --test-path ./data_test/bird_rgba.png
``` -->

## Testing

```bash
python test.py stereogs \
    --resume ./ckpts/model.safetensors \
    --workspace ./workspace_eval \
    --eval-views-path-json /path/to/eval_dataset.json
```

## Available Configs

| Config | Input Views | UNet Refinement | Description |
|--------|-------------|-----------------|-------------|
| `stereogs` | 4 | Yes | Default 4-view with UNet |
| `stereogs_6v` | 6 | Yes | 6-view with UNet |
| `stereogs_no_refinement` | 4 | No | Direct Gaussian prediction |
| `stereogs_6v_no_refinement` | 6 | No | 6-view direct Gaussian |

## Project Structure

```
stereo-gs/
├── core/
│   ├── stereogs.py      # Main StereoGS model
│   ├── options.py       # Configuration options
│   ├── gs.py            # Gaussian splatting renderer
│   ├── unet.py          # UNet refinement network
│   └── utils.py         # Utilities
├── dust3r/              # DUST3R encoder
├── mast3r/              # MASt3R stereo matching
├── croco/               # CroCo backbone
├── mvdream/             # MVDream multi-view generation
├── infer.py             # 4-view inference
├── infer_v3d.py         # N-view inference
├── infer_mvdream.py     # MVDream inference
└── eval_stereogs.py     # Evaluation script
```

## Acknowledgement

This project is based on [MASt3R](https://github.com/naver/mast3r) and [LGM](https://github.com/3DTopia/LGM).
