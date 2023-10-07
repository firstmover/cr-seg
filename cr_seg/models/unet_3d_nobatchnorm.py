import typing as tp

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.registry import MODELS

from cr_seg.criterion import build_loss


@MODELS.register_module()
class UNet3dDropout(BaseModel):
    def __init__(
        self,
        model_size: str = "large",
        dropout: bool = True,
        loss_cfg: tp.Optional[dict] = None,
    ):
        super(UNet3dDropout, self).__init__()
        self.model = UNet(size=model_size, dropout=dropout)

        if loss_cfg is not None:
            self.criterion = build_loss(loss_cfg)

    def forward(self, img, gt_seg_map, mode):
        outputs = self.model(img)
        if mode == "loss":
            loss = self.criterion(outputs, gt_seg_map)
            return {"loss": loss}
        elif mode == "predict":
            preds = torch.argmax(outputs, dim=1)
            return preds
        elif mode == "predict_and_logits":
            preds = torch.argmax(outputs, dim=1)
            return preds, outputs


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        batch_norm=True,
        squeeze=False,
        dropout=True,
        size: str = "small",
    ):
        super(UNet, self).__init__()

        if size == "large":
            self.conv1 = Conv(in_channels, 64, batch_norm=batch_norm, dropout=dropout)
            self.down1 = Down(64, 128, batch_norm=batch_norm, dropout=dropout)
            self.down2 = Down(128, 256, batch_norm=batch_norm, dropout=dropout)
            self.down3 = Down(256, 512, batch_norm=batch_norm, dropout=dropout)
            self.down4 = Down(512, 512, batch_norm=batch_norm, dropout=dropout)
            self.up1 = Up(512, 256, batch_norm=batch_norm, dropout=dropout)
            self.up2 = Up(256, 128, batch_norm=batch_norm, dropout=dropout)
            self.up3 = Up(128, 64, batch_norm=batch_norm, dropout=dropout)
            self.up4 = Up(64, 64, batch_norm=batch_norm, dropout=dropout)
            self.out = OutConv(64, out_channels)

        elif size == "medium":
            self.conv1 = Conv(in_channels, 32, batch_norm=batch_norm, dropout=dropout)
            self.down1 = Down(32, 64, batch_norm=batch_norm, dropout=dropout)
            self.down2 = Down(64, 128, batch_norm=batch_norm, dropout=dropout)
            self.down3 = Down(128, 256, batch_norm=batch_norm, dropout=dropout)
            self.down4 = Down(256, 256, batch_norm=batch_norm, dropout=dropout)
            self.up1 = Up(256, 128, batch_norm=batch_norm, dropout=dropout)
            self.up2 = Up(128, 64, batch_norm=batch_norm, dropout=dropout)
            self.up3 = Up(64, 32, batch_norm=batch_norm, dropout=dropout)
            self.up4 = Up(32, 32, batch_norm=batch_norm, dropout=dropout)
            self.out = OutConv(32, out_channels)

        elif size == "small":
            self.conv1 = Conv(in_channels, 16, batch_norm=batch_norm, dropout=dropout)
            self.down1 = Down(16, 32, batch_norm=batch_norm, dropout=dropout)
            self.down2 = Down(32, 64, batch_norm=batch_norm, dropout=dropout)
            self.down3 = Down(64, 128, batch_norm=batch_norm, dropout=dropout)
            self.down4 = Down(128, 128, batch_norm=batch_norm, dropout=dropout)
            self.up1 = Up(128, 64, batch_norm=batch_norm, dropout=dropout)
            self.up2 = Up(64, 32, batch_norm=batch_norm, dropout=dropout)
            self.up3 = Up(32, 16, batch_norm=batch_norm, dropout=dropout)
            self.up4 = Up(16, 16, batch_norm=batch_norm, dropout=dropout)
            self.out = OutConv(16, out_channels)

        else:
            raise ValueError('size must be one of "small", "medium", or "large"')

        self.squeeze = squeeze
        # self.batch_norm = batch_norm

    def forward(self, x):
        if self.squeeze:
            x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        if self.squeeze:
            x = x.squeeze(1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, dropout=False):
        super(Conv, self).__init__()
        if dropout:
            dropout_p = 0.05
        else:
            dropout_p = 0
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_p),
                nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_p),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_p),
                nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_p),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, dropout=False):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            Conv(in_size, out_size, batch_norm=batch_norm, dropout=dropout),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, dropout=False):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=2, stride=2)
        self.conv = Conv(in_size * 2, out_size, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x1, x2):
        up = self.up(x1)
        out = torch.cat([up, x2], dim=1)
        out = self.conv(out)
        return out
