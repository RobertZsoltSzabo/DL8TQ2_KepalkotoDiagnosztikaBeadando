from torch import nn


class Generator(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, feature_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_channels * 8, feature_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( feature_channels * 4, feature_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( feature_channels * 2, feature_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( feature_channels, feature_channels//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_channels//2, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_channels, input_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, feature_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=feature_channels*2*(input_size//4)*(input_size//4), out_features=1, bias=False),
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