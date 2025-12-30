# Stereo-GS

Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction (ACM MM 2025)

[Paper](https://arxiv.org/abs/2507.14921) | [Project](https://kevinhuangxf.github.io/stereo-gs/)

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

The download script will automatically put the checkpoints and data to the target folder.

```bash
python download.py --with-data
```

## Inference

### Inference V3D

Please reference the [V3D](https://github.com/heheyas/V3D) for generating a sequency of multi-view images. We provide an example under the [data/basketball](data/basketball) path.

**4 views:**
```bash
python infer_v3d.py stereogs \
    --resume ./ckpts/model.safetensors \
    --workspace ./output/basketball \
    --test-path ./data/basketball
```

**6 views:**
```bash
python infer_v3d.py stereogs_6v \
    --resume ./ckpts/model.safetensors \
    --workspace ./output/basketball_6v \
    --test-path ./data/basketball
```

The `--test-path` should point to a folder containing the [V3D](https://github.com/heheyas/V3D) generated images. The script will auto-detect and select evenly-spaced images.

### Inference with MVDream

Generate multi-view images from a single RGBA image:

```bash
python infer_mvdream.py stereogs \
    --resume ./ckpts/model.safetensors \
    --workspace ./output/bubble_mart_blue \
    --test-path ./data/bubble_mart_blue.png
```

## Testing

```bash
python test.py stereogs \
    --resume ./ckpts/model.safetensors \
    --workspace ./workspace_eval \
    --eval-views-path-json ./data/gso_test_4v.json
```

<!-- ## Available Configs

| Config | Input Views | UNet Refinement | Description |
|--------|-------------|-----------------|-------------|
| `stereogs` | 4 | Yes | Default 4-view with UNet |
| `stereogs_6v` | 6 | Yes | 6-view with UNet |
| `stereogs_no_refinement` | 4 | No | Direct Gaussian prediction |
| `stereogs_6v_no_refinement` | 6 | No | 6-view direct Gaussian | -->

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
├── infer_v3d.py         # V3D inference
├── infer_mvdream.py     # MVDream inference
└── test.py              # Testing script
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2025stereogs,
      title     = {Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction},
      author    = {Xiufeng Huang, Ka Chun Cheung, Runmin Cong, Simon See, Renjie Wan},
      booktitle = {ACM Multimedia},
      year      = {2025}
}
```

## Acknowledgement

This project is based on [MASt3R](https://github.com/naver/mast3r) and [LGM](https://github.com/3DTopia/LGM).
