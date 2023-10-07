#################
#  environment  #
#################

find_unused_parameters = False
sync_bn = "torch"

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

###############################
#  logging and visualization  #
###############################

work_dir = None  # specify in command line arg

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(type="TensorboardVisBackend"),
    ],
    name="visualizer",
)

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    logger=dict(type="LoggerHook", interval=16),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best="regi_mse/val",
        rule="less",
    ),
    check_invalid_loss=dict(type="CheckInvalidLossHook"),
)
