import math
import numpy as np
from torch import nn
from torch.nn import functional as F 


class mf_mod(nn.Module):
  def __init__(self, in_channels, out_channels, pool_size,n_parallels=6):
    super(mf_mod, self).__init__()
    self.pooling = nn.Sequential(nn.AvgPool2d(kernel_size=(pool_size,1)),
                                              nn.BatchNorm2d(in_channels))
    convs = []
    filter = 32 
    for i in range(n_parallels):
      convs.append(nn.Conv2d(in_channels=in_channels, out_channels=26, kernel_size=(1, filter), padding=filter//2))
      filter += 32

    self.conv_modules = nn.ModuleList(convs)
    self.bottleneck = nn.Conv2d(in_channels=26, out_channels=out_channels, kernel_size=1)
    self.elu = nn.ELU()

  def forward(self, x):
    output = self.pooling(x)
    parallel_output = []
    for conv in self.conv_modules:
      parallel_output.append(self.elu(conv(output)))

    output = torch.cat(parallel_output, dim=1)
    output = self.elu(self.bottleneck(output))
    return output


class TempoCNN(nn.Module):
  def __init__(self, kernel_size=5, n_blocks=4, mf_channels=36, dropout=0.5):
    super(TempoCNN,self).__init__()
    main = [nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,kernel_size)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,kernel_size)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,kernel_size)),
            nn.ELU(),
            nn.BatchNorm2d(16)]


    main += [mf_mod(in_channels=16, out_channels=mf_channels, pool_size=5)]
    for i in range(n_blocks-2):
      main += [mf_mod(in_channels=mf_channels, out_channels=mf_channels, pool_size=2)]
    
    main += [mf_mod(in_channels=mf_channels, out_channels=1, pool_size=2)]
    main += [nn.BatchNorm2d(64), #check channel
             nn.Dropout(dropout),
             nn.Linear(64,64),
             nn.ELU(),
             nn.BatchNorm2d(64),
             nn.Linear(64,64),
             nn.ELU(),
             nn.BatchNorm2d(64),
             nn.Linear(64,256),
             nn.Softmax()
             ]

    self.main = nn.Sequential(main)

  def forward(self, x):
    return self.main(x)