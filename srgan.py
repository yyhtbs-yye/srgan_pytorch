import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):

        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)

        z += x
        return z

class Enlarger_PixelShuffle(nn.Module):

    def __init__(self):
        super(Enlarger_PixelShuffle, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(16)])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual = x

        x = self.residual_blocks(x)
        
        x = self.bn1(self.conv2(x))
        x += residual

        x = self.up1(x)
        x = self.up2(x)

        x = self.conv3(x)
        return torch.tanh(x)
    
class Enlarger_Bilinear(nn.Module):
    def __init__(self):
        super(Enlarger_Bilinear, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(16)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.residual_blocks(x)
        x = self.bn1(self.conv2(x))
        x += identity
        x = self.upsample1(x)
        x = self.bn2(self.conv3(x))
        x = self.upsample2(x)
        x = self.bn3(self.conv4(x))
        x = self.conv5(x)
        return self.tanh(x)


class Discriminator_ResCon(nn.Module):

    def __init__(self, dim=64):
        super(Discriminator_ResCon, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim * 2)

        self.conv3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim * 4)

        self.conv4 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim * 8)

        self.conv5 = nn.Conv2d(dim * 8, dim * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(dim * 16)

        self.conv6 = nn.Conv2d(dim * 16, dim * 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(dim * 32)

        self.conv7 = nn.Conv2d(dim * 32, dim * 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(dim * 16)

        self.conv8 = nn.Conv2d(dim * 16, dim * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(dim * 8)

        # Residual connection
        self.conv9 = nn.Conv2d(dim * 8, dim * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(dim * 2)

        self.conv10 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(dim * 2)

        self.conv11 = nn.Conv2d(dim * 2, dim * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(dim * 8)

        self.flat = nn.Flatten()
        self.dense = nn.LazyLinear(1)  # Update (size after convs) appropriately

    def forward(self, x):
        res = self.lrelu(self.conv1(x))

        x = self.lrelu(self.bn1(self.conv2(res)))
        x = self.lrelu(self.bn2(self.conv3(x)))
        x = self.lrelu(self.bn3(self.conv4(x)))
        x = self.lrelu(self.bn4(self.conv5(x)))
        x = self.lrelu(self.bn5(self.conv6(x)))
        x = self.lrelu(self.bn6(self.conv7(x)))
        x = self.lrelu(self.bn7(self.conv8(x)))

        temp = x

        x = self.lrelu(self.bn8(self.conv9(x)))
        x = self.lrelu(self.bn9(self.conv10(x)))
        x = self.lrelu(self.bn10(self.conv11(x)))

        x += temp  # Applying the residue connection
        x = self.flat(x)
        x = self.dense(x)

        return x

class Discriminator_Naive(nn.Module):
    """
    Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.
    Feature maps (n) and stride (s) feature maps (n) and stride (s).
    """
    def __init__(self):
        super(Discriminator_Naive, self).__init__()
        # Initialize layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.flat = nn.Flatten()
        self.dense1 = nn.LazyLinear(1024)  # Update the dimensions according to your input size
        self.dense2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.bn1(self.conv2(x)))
        x = self.lrelu(self.bn2(self.conv3(x)))
        x = self.lrelu(self.bn3(self.conv4(x)))
        x = self.lrelu(self.bn4(self.conv5(x)))
        x = self.lrelu(self.bn5(self.conv6(x)))
        x = self.lrelu(self.bn6(self.conv7(x)))
        x = self.lrelu(self.bn7(self.conv8(x)))
        x = self.flat(x)
        x = self.lrelu(self.dense1(x))
        logits = self.dense2(x)
        n = self.sigmoid(logits)
        return n
