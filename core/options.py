import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1 # 67.38 # 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # 2.0 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    # total views
    total_views_number: int = 32

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 1
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'fp16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None
    local_models_path_json: Optional[str] = None
    local_views_path_json: str = '/workspace/code/objaverse-rendering/valid_views.json'
    eval_views_path_json: str = '/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json'
    camera_augmentation: bool = False

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    ### additional settings
    use_depth_net: bool = False
    use_depth_offset: bool = False
    use_depth: bool = False
    use_geometry_net: bool = False

    # log_steps
    log_steps: int = 100

    # scaling dataset length
    dataset_scalar: float = 1.0

@dataclass
class Options_StereoGS:
    ### model
    model_type = "StereoGS"
    # Unet image input size
    input_size: int = 256
    # Unet definition (from mast3r)
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256
    # use mv_head
    mv_head: bool = False
    # use_unet_refinement: True = MASt3R desc → UNet → Gaussians; False = MASt3R head → Gaussians directly
    use_unet_refinement: bool = True

    ### dataset
    # data mode
    data_mode: Literal['s3'] = 'dust3r'
    # fovy of the dataset
    fovy: float = 67.38
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    # number of input views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5
    # num workers
    num_workers: int = 8
    # depth data
    use_depth: bool = False
    # local_views_path
    local_views_path_json: Optional[str] = None
    # eval_views_path
    eval_views_path_json: Optional[str] = None
    # local_models_path
    local_models_path_json: Optional[str] = None
    # ply_files_path
    ply_files_path: Optional[str] = None
    # camera augmentation
    camera_augmentation: bool = False

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 1
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # chamfer loss weight
    lambda_chamfer: float = 1.0
    # depth loss weight
    lambda_depth: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'fp16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5
    # log_steps
    log_steps: int = 100

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    # fovy=67.38/2,
    fovy=49.1,
    use_depth=True,
    output_size=256, # render & supervise Gaussians at a higher resolution.
    batch_size=1,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['stereogs'] = 'StereoGS config with UNet refinement'
config_defaults['stereogs'] = Options_StereoGS(
    input_size=256,
    output_size=256,  # For UNet mode; Direct Gaussian mode uses 128
    num_input_views=4,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16', # fp16 or bf16
    num_epochs=100,
)

# Direct Gaussian mode config (matches mast3rgs checkpoint)
config_doc['stereogs_no_refinement'] = 'StereoGS Direct Gaussian mode - no UNet refinement'
config_defaults['stereogs_no_refinement'] = Options_StereoGS(
    input_size=256,
    output_size=128,
    mv_head=True,
    use_unet_refinement=False,  # Direct Gaussian mode
    num_input_views=4,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16', # fp16 or bf16
    num_epochs=100,
)

# StereoGS 6-view config
config_doc['stereogs_6v'] = 'StereoGS config with 6 input views'
config_defaults['stereogs_6v'] = Options_StereoGS(
    input_size=256,
    output_size=256,
    num_input_views=6,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=100,
)

# StereoGS 6-view config without UNet refinement
config_doc['stereogs_6v_no_refinement'] = 'StereoGS 6-view Direct Gaussian mode'
config_defaults['stereogs_6v_no_refinement'] = Options_StereoGS(
    input_size=256,
    output_size=128,
    mv_head=True,
    use_unet_refinement=False,
    num_input_views=6,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=100,
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
