# just give the entire image to the model
# this works for non Transformer models
tta_input_size = (112, 112, 80)
tta_model = dict(
    type="UNet3dDropoutInference",
    inference_cfg=dict(type="AllCrop"),
    model_size="medium",
    loss_cfg=None,
)
