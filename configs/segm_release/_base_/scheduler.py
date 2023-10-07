#############################
#  optimizer and scheduler  #
#############################

total_epoch = 500 // 5
warmup_epoch = 10 // 5

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=warmup_epoch,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=total_epoch - warmup_epoch,
        by_epoch=True,
        begin=warmup_epoch,
        end=total_epoch,
        convert_to_iter_based=True,
    ),
]

base_learning_rate = 0.0005 * 2  # because batch size got * 2
weight_decay = 0.0001
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        type="Adam",
        lr=base_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    ),
)

val_evaluator = dict(type="DiceMetric", eval_by_split=True)
