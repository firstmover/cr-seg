_base_ = [
    "./_base_/dataset.py",
    "./_base_/scheduler.py",
    "./_base_/default_runtime_lambda_schedule.py",
]

total_epoch = {{_base_.total_epoch}}
train_paired_unlabeled_dataloader = {{_base_.train_paired_unlabeled_dataloader}}

# train_dataloader is by default passed to train loop builder
# using the key "dataloader"
train_cfg = dict(
    type="LabeledUnlabeledEpochBasedTrainLoop",
    unlabeled_dataloader=train_paired_unlabeled_dataloader,
    max_epochs=total_epoch,
    val_interval=1,
)
val_cfg = dict()

loss = dict(
    type="DiceCELoss",
    softmax=True,
    to_onehot_y=True,
    squared_pred=True,
    include_background=True,
)

crop_size = (80, 80, 64)
result_root = "{{update this path to where you save your results}}"
model = dict(
    type="UNet3dCRWithRegiv2",
    model_size="medium",
    dropout=False,
    loss_cfg=loss,
    lambda_schedule_cfg=dict(
        type="ExpRampUpConstantSchedule", value=1.0, ramp_up_total_ratio=0.5
    ),
    lambda_t_schedule_cfg=dict(
        type="ExpRampUpConstantSchedule", value=1.0, ramp_up_total_ratio=0.5
    ),
    teacher_momentum=1 - 0.999,  # convention in mmengine
    cr_trans_cfg=[
        dict(type="RandomRot903d"),
        dict(type="RamdomFlip3d"),
        dict(type="RandomRot3d"),
        dict(type="RandomTranslation3d", max_shift_pix=(5, 5, 5), p=0.1),
        dict(type="RandomGamma", gamma_range=(0.5, 2), p=0.5),
        dict(type="RandomGaussianNoise", sigma_range=(0.0, 0.1), p=0.1),
    ],
    registration_cfg=dict(
        type="VoxelMorph",
        inshape=crop_size,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=result_root
            + "placenta/med_img_ssl/regi_release/voxelmorph/exp_5_0/epoch_25.pth",
        ),
    ),
)
