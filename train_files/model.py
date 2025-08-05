import torch
import torch.nn as nn

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conditional_UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, c_embd=1024):
        super(Conditional_UNet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)

        self.bottleneck = double_conv(512, c_embd)

        # Upsample layers
        self.up1 = nn.ConvTranspose2d(c_embd, 512, 2, stride=2)
        self.up_conv1 = double_conv(c_embd, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv2 = double_conv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv3 = double_conv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv4 = double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, out_ch, 1)
        self.color_fc = nn.Linear(out_ch, c_embd)

    def init_weight(self, std=0.2):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0. if isinstance(m, nn.Conv2d) else 1., std)
                if isinstance(m, nn.Linear):
                    m.bias.data.fill_(0)

    def forward(self, x, color, c_embd=1024):
        c1 = self.dconv_down1(x)
        p1 = self.maxpool(c1)

        c2 = self.dconv_down2(p1)
        p2 = self.maxpool(c2)

        c3 = self.dconv_down3(p2)
        p3 = self.maxpool(c3)

        c4 = self.dconv_down4(p3)
        p4 = self.maxpool(c4)

        bottleneck = self.bottleneck(p4)

        # Color conditioning
        color_embedding = self.color_fc(color).view(-1, c_embd, 1, 1)
        bottleneck = bottleneck + color_embedding

        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, c4], dim=1)
        u1 = self.up_conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, c3], dim=1)
        u2 = self.up_conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, c2], dim=1)
        u3 = self.up_conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, c1], dim=1)
        u4 = self.up_conv4(u4)

        output = self.final_conv(u4)
        return torch.sigmoid(output)
