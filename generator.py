import torch
from attention import SpatialTransformer
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, num_heads=1, context_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, num_filters[0], kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.InstanceNorm2d(num_filters[0])
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(num_filters[1])
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(num_filters[2])
        self.act3 = nn.ReLU(inplace=True)

        self.cross_attention = SpatialTransformer(num_filters[2], num_heads, num_filters[2] // num_heads, depth=1, context_dim=context_dim)

        self.deconv1 = nn.ConvTranspose2d(num_filters[2], num_filters[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dnorm1 = nn.InstanceNorm2d(num_filters[1])
        self.dact1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(num_filters[1], num_filters[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dnorm2 = nn.InstanceNorm2d(num_filters[0])
        self.dact2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.Conv2d(num_filters[0], output_dim, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x, cond):
        # Encode, cross-attend on text, then decode to a perturbation.
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        cond = cond.unsqueeze(1)
        x = self.cross_attention(x, cond)
        x = self.dact1(self.dnorm1(self.deconv1(x)))
        x = self.dact2(self.dnorm2(self.deconv2(x)))
        x = self.tanh(self.deconv3(x))
        return x
