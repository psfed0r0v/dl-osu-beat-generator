import torch
from torch import nn
from config import get_params

params = get_params()


class MultiFilterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, n_parallels=4):
        super(MultiFilterBlock, self).__init__()
        self.pooling = nn.Sequential(nn.AvgPool2d(kernel_size=(pool_size, 1)),
                                     nn.BatchNorm2d(in_channels))
        convs = []
        filter = 4
        for i in range(n_parallels):
            convs.append(nn.Conv2d(in_channels=in_channels, out_channels=26,
                                   kernel_size=(1, filter), padding=filter // 2))
            filter += 4

        self.conv_modules = nn.ModuleList(convs)
        self.bottleneck = nn.Conv2d(in_channels=26, out_channels=out_channels, kernel_size=1)
        self.elu = nn.ELU()

    def forward(self, x):
        output = self.pooling(x)
        parallel_output = []
        for conv in self.conv_modules:
            parallel_output.append(self.elu(conv(output)))

        output = torch.cat(parallel_output, dim=2)

        output = self.elu(self.bottleneck(output))
        return output


class TempoCNN(nn.Module):
    def __init__(self, kernel_size=5, n_blocks=3, mf_channels=24, dropout=0.2):
        super(TempoCNN, self).__init__()
        main = [nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, kernel_size),
                          padding=(0, kernel_size // 2)),
                nn.ELU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, kernel_size),
                          padding=(0, kernel_size // 2)),
                nn.ELU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, kernel_size),
                          padding=(0, kernel_size // 2)),
                nn.ELU(),
                nn.BatchNorm2d(16)]

        main += [MultiFilterBlock(in_channels=16, out_channels=mf_channels, pool_size=6)]
        for i in range(n_blocks - 1):
            main += [MultiFilterBlock(in_channels=mf_channels, out_channels=mf_channels, pool_size=6)]

        main += [nn.BatchNorm2d(mf_channels),
                 nn.Conv2d(in_channels=mf_channels, out_channels=1, kernel_size=1),
                 nn.MaxPool2d(4),
                 nn.Flatten(),
                 nn.Dropout(dropout),
                 nn.Linear(7176, 4112),
                 nn.ELU(),
                 nn.BatchNorm1d(4112),
                 nn.Linear(4112, params.OUT_SHAPE),
                 #  nn.ELU(),
                 #  nn.BatchNorm1d(1028),
                 #  nn.Linear(1028,5001),
                 nn.Sigmoid()]

        self.net = nn.Sequential(*main)

    def forward(self, x):
        return self.net(x)
