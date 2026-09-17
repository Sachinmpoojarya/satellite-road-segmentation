import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResNetUNet(nn.Module):
    """
    U-Net with Pre-trained ResNet34 Encoder Backbone.
    Dramatically improves feature extraction for low-grade, unpaved, and rural roads.
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(ResNetUNet, self).__init__()

        # Pretrained ResNet34 encoder
        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
        except Exception:
            resnet = models.resnet34(pretrained=pretrained)

        # Encoder stages
        self.encoder_layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # (64, H/2, W/2)
        self.maxpool = resnet.maxpool # (64, H/4, W/4)
        self.encoder_layer1 = resnet.layer1 # (64, H/4, W/4)
        self.encoder_layer2 = resnet.layer2 # (128, H/8, W/8)
        self.encoder_layer3 = resnet.layer3 # (256, H/16, W/16)
        self.encoder_layer4 = resnet.layer4 # (512, H/32, W/32)

        # Decoder stages with skip connections
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        e0 = self.encoder_layer0(x)     # (64, H/2, W/2)
        p0 = self.maxpool(e0)           # (64, H/4, W/4)
        e1 = self.encoder_layer1(p0)    # (64, H/4, W/4)
        e2 = self.encoder_layer2(e1)    # (128, H/8, W/8)
        e3 = self.encoder_layer3(e2)    # (256, H/16, W/16)
        e4 = self.encoder_layer4(e3)    # (512, H/32, W/32)

        # Decoder forward pass with skip connections
        d4 = self.upconv4(e4)           # (256, H/16, W/16)
        d4 = torch.cat([d4, e3], dim=1)    # (512, H/16, W/16)
        d4 = self.dec4(d4)              # (256, H/16, W/16)

        d3 = self.upconv3(d4)           # (128, H/8, W/8)
        d3 = torch.cat([d3, e2], dim=1)    # (256, H/8, W/8)
        d3 = self.dec3(d3)              # (128, H/8, W/8)

        d2 = self.upconv2(d3)           # (64, H/4, W/4)
        d2 = torch.cat([d2, e1], dim=1)    # (128, H/4, W/4)
        d2 = self.dec2(d2)              # (64, H/4, W/4)

        d1 = self.upconv1(d2)           # (64, H/2, W/2)
        d1 = torch.cat([d1, e0], dim=1)    # (128, H/2, W/2)
        d1 = self.dec1(d1)              # (64, H/2, W/2)

        d0 = self.upconv0(d1)           # (32, H, W)
        d0 = self.dec0(d0)              # (32, H, W)

        out = self.conv_final(d0)       # (1, H, W)
        return torch.sigmoid(out)

# Maintain full backwards compatibility across the project
UNet = ResNetUNet

