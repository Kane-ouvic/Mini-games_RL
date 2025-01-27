import torch
import torch.nn as nn

# Basic Block
class Basic_C2D_Block(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, stride, is_BN):
        super(Basic_C2D_Block, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=k_size, stride=stride, padding=k_size // 2
        )
        self.bn_1 = nn.BatchNorm2d(out_dim) if is_BN else nn.Identity()              
        self.lrelu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        y = self.conv_1(x)
        y = self.bn_1(y)
        return self.lrelu(y)

# Residual Block
class Res_C2D_Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, stride=1):
        super(Res_C2D_Block, self).__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(
                Basic_C2D_Block(
                    in_dim=in_dim if i == 0 else out_dim,
                    out_dim=out_dim,
                    k_size=3,
                    stride=stride if i == 0 else 1, 
                    is_BN=False,
                )
            )
        self.blocks = nn.Sequential(*layers)

        self.adjust_residual = None
        if in_dim != out_dim or stride != 1:
            self.adjust_residual = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x):
        residual = x
        if self.adjust_residual:
            residual = self.adjust_residual(x)

        y = self.blocks(x)
        y += residual
        return nn.LeakyReLU(inplace=False)(y)
    

# Squeeze-and-Excitation Block
class SE_Block(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super(SE_Block, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.global_avg_pool(x).view(x.size(0), -1)
        scale = self.fc(scale).view(x.size(0), x.size(1), 1, 1)
        return x * scale

class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CustomCNN, self).__init__()

        channels, _, _ = input_shape

        self.basic = Basic_C2D_Block(channels, 24, k_size=4, stride=4, is_BN=False)   # Basic_C2D_Block
        self.res1  = Res_C2D_Block(24, 48, num_blocks=2, stride=2)                    # Res_C2D_Block
        self.res2  = Res_C2D_Block(48, 96, num_blocks=2, stride=2)                    # Res_C2D_Block

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)                                # Adaptive Global Average Pooling (自適應全局平均池化)
        self.fc = nn.Linear(96, num_actions)                                          # Fully Connected Layer (全連接層)

    def forward(self, x):
        x = self.basic(x)
        x = self.res1(x)
        x = self.res2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    

# Improved CNN Model
class ImprovedCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ImprovedCNN, self).__init__()
        channels, _, _ = input_shape

        self.basic = Basic_C2D_Block(channels, 32, k_size=3, stride=2, is_BN=True)
        self.res1 = Res_C2D_Block(32, 64, num_blocks=3, stride=2)
        self.res2 = Res_C2D_Block(64, 128, num_blocks=3, stride=2)
        self.se = SE_Block(128)  # Attention mechanism

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.basic(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.se(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

