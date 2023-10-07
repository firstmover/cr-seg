crop_size = (80, 80, 64)
train_transform = [
    dict(type="RandomFlip3d", dim=0),
    dict(type="RandomFlip3d", dim=1),
    dict(type="RandomFlip3d", dim=2),
    dict(type="RandomRot903d"),
    dict(
        type="Crop3D", mode="random", crop_size=crop_size, pad_zero_to_match_shape=True
    ),
    dict(type="DefaultFormatBundle3D"),
    dict(type="Collect", keys=["img", "gt_seg_map"], meta_keys=[]),
]
validation_transform = [
    dict(
        type="Crop3D", mode="center", crop_size=crop_size, pad_zero_to_match_shape=True
    ),
    dict(type="DefaultFormatBundle3D"),
    dict(type="Collect", keys=["img", "gt_seg_map"], meta_keys=["split_tag"]),
]

train_dataset = dict(
    type="placenta_3d_registration",
    train=True,
    n_repeat_train=1,
    transform=train_transform,
    pairing_pattern="all",
    num_fold=5,
    val_fold_idx=0,
)
val_dataset = dict(
    type="placenta_3d_registration",
    train=False,
    transform=validation_transform,
    validation_sub_sample_rate=0.05,
    pairing_pattern="consecutive",
    num_fold=5,
    val_fold_idx=0,
)

batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="MaxNumSampler", shuffle=True, max_num_samples=1024),
    collate_fn=dict(type="default_collate"),
    dataset=train_dataset,
    persistent_workers=True,
    pin_memory=True,
    num_workers=8,
)
val_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
    dataset=val_dataset,
    persistent_workers=True,
    pin_memory=True,
    num_workers=8,
)
