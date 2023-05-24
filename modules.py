from torch import nn

# Insert BasicConv2d

#layer for ofpn
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(Conv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.BatchNorm2d(2, out_channels)
        self.act = nn.ReLU(0.2)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(Conv3d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.BatchNorm3d(2, out_channels)
        self.act = nn.ReLU(0.2)# ask about in place

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

# Insert ConvSC


# Insert GroupConv2d


# Insert Inception

