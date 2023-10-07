crop_size = (80, 80, 64)

train_transform = [
    dict(type="RandomBiasFieldMultiChan", same_all_channels=True),
    dict(type="RandomFlip3d", dim=0),
    dict(type="RandomFlip3d", dim=1),
    dict(type="RandomFlip3d", dim=2),
    dict(type="RandomRot903d"),
    dict(type="RandomGammaAdjustContrastMultiChan", same_all_channels=True),
    dict(type="RandomRotate3DSingleDim", prob=0.5),
    dict(type="RandomCrop3D", crop_size=crop_size),
    dict(type="DefaultFormatBundle3D"),
    dict(type="Collect", keys=["img", "gt_seg_map"], meta_keys=[]),
]
train_paired_unlabeled_transform = train_transform
validation_transform = [
    dict(type="CenterCrop3D", crop_size=crop_size, pad_zero_to_match_shape=True),
    dict(type="DefaultFormatBundle3D"),
    dict(type="Collect", keys=["img", "gt_seg_map"], meta_keys=["split_tag"]),
]

train_dataset = dict(
    type="placenta_3d_trimed_normed_cross_validation_for_cr",
    train=True,
    n_repeat_train=10 // 2 * 5,
    norm_method="linear_0_99_percentile",
    transform=train_transform,
    num_fold=5,
    val_fold_idx=0,
    include_unlabeled=False,
    use_raw=True,
    corrected_label_tag="split-nifti-raw-revised-2023-06-06",
)
# the placenta_3d_registration dataset will yield paired unlabeled data
# could be randomly paired for sequentially paired
train_paired_unlabeled_dataset = dict(
    type="placenta_3d_registration",
    train=True,
    n_repeat_train=10 // 2 * 5,
    norm_method="linear_0_99_percentile",
    transform=train_paired_unlabeled_transform,
    num_fold=5,
    val_fold_idx=0,
    use_raw=True,
    corrected_label_tag="split-nifti-raw-revised-2023-06-06",
    pairing_pattern="consecutive",
)
val_dataset = dict(
    type="placenta_3d_trimed_normed_cross_validation_for_cr_with_tc",
    train=False,
    n_repeat_train=1,
    norm_method="linear_0_99_percentile",
    transform=validation_transform,
    num_fold=5,
    val_fold_idx=0,
    include_unlabeled=True,
    use_raw=True,
    corrected_label_tag="split-nifti-raw-revised-2023-06-06",
)

batch_size = 4  # effectively sample 4 labeled and 4 pairs of unlabeled
num_workers = 8
train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="default_collate"),
    dataset=train_dataset,
    persistent_workers=True,
    pin_memory=True,
    num_workers=num_workers,
)
# we will take the max of label and unlabeled iterator
# for epoch based training to stop iterating
train_paired_unlabeled_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="default_collate"),
    dataset=train_paired_unlabeled_dataset,
    persistent_workers=True,
    pin_memory=True,
    num_workers=num_workers,
)
val_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
    dataset=val_dataset,
    num_workers=num_workers,
)
