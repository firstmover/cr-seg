_base_ = [
    "./_base_/dataset.py",
    "./_base_/scheduler.py",
    "./_base_/default_runtime.py",
]

crop_size = (80, 80, 64)
model = dict(
    type="VoxelMorphSegMask",
    inshape=crop_size,
    use_seg_mask=False,
    loss_cfg=dict(type="MSE", return_loss_map=True),
)
