from torch import nn

# Generator Code

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super(UpsamplingBlock, self).__init__()
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   bias=bias)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, input):
        return self.activation(self.batchnorm(self.conv2d_transpose(input)))
    

class Generator2(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels):
        super(Generator2, self).__init__()
        self.network = nn.Sequential(
            UpsamplingBlock(in_channels=in_channels,
                            out_channels=feature_channels*8,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            bias=False), # out: 4*4
            UpsamplingBlock(in_channels=feature_channels*8,
                            out_channels=feature_channels*4,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False), # out: 8*8
            UpsamplingBlock(in_channels=feature_channels*4,
                            out_channels=feature_channels*2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False), # out: 16*16
            UpsamplingBlock(in_channels=feature_channels*2,
                            out_channels=feature_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False), # out: 32*32
            nn.ConvTranspose2d(in_channels=feature_channels,
                               out_channels=out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False), # out: 64*64
            nn.Tanh()
        )
        weights_init(self)

    def forward(self, input):
        return self.network(input)



class Generator(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, feature_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(feature_channels * 8, feature_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( feature_channels * 4, feature_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( feature_channels * 2, feature_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( feature_channels, feature_channels//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(nc) x 64 x 64`'
            nn.ConvTranspose2d(feature_channels//2, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.main(input)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, batchnorm=True):
        super(DownsamplingBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        if self.batchnorm:
            return self.activation(self.batchnorm(self.conv2d(input)))
        else:
            return self.activation(self.conv2d(input))


class Discriminator2(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super(Discriminator2, self).__init__()
        self.network = nn.Sequential(
            DownsamplingBlock(in_channels=in_channels,
                              out_channels=feature_channels,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False,
                              batchnorm=False),
            DownsamplingBlock(in_channels=feature_channels,
                              out_channels=feature_channels*2,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False),
            DownsamplingBlock(in_channels=feature_channels*2,
                              out_channels=feature_channels*4,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False),
            DownsamplingBlock(in_channels=feature_channels*4,
                              out_channels=feature_channels*8,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False),
            nn.Conv2d(in_channels=feature_channels*8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.network(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_channels, input_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(in_channels, feature_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(feature_channels, feature_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. ``(ndf*2) x 16 x 16``
            # state size. ``(ndf*8) x 4 x 4``
            #nn.Conv2d(feature_channels * 2, 1, input_size//4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(in_features=feature_channels*2*(input_size//4)*(input_size//4), out_features=1, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
    
class DiscriminatorOriginal(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super(DiscriminatorOriginal, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(in_channels, feature_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(feature_channels, feature_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(feature_channels * 2, feature_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(feature_channels * 4, feature_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(feature_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
    

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(tensor=model.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(tensor=model.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(tensor=model.bias.data, val=0)