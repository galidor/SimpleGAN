import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ndf=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, ndf, 4, 2, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, 2*ndf, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2*ndf)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(2*ndf, 4*ndf, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*ndf)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(4*ndf, 8*ndf, 4, 2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(8*ndf)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(8*ndf, 1, 4, 1, 0, bias=False)
        self.sigmoid5 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.sigmoid5(x)
        return x


class DiscriminatorSmall(nn.Module):
    def __init__(self, ndf):
        super(DiscriminatorSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, ndf, 4, 2, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, 2*ndf, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2*ndf)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(2*ndf, 4*ndf, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*ndf)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(4*ndf, 1, 3, 1, 0, bias=False)
        self.sigmoid4 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        # x = self.sigmoid4(x)
        return x


class Generator(nn.Module):
    def __init__(self, ngf):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, 8*ngf, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8*ngf)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(8*ngf, 4*ngf, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*ngf)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(4*ngf, 2*ngf, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2*ngf)
        self.relu3 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(2*ngf, ngf, 4, 2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv5 = nn.ConvTranspose2d(ngf, 1, 4, 2, padding=1, bias=False)
        self.relu5 = nn.Tanh()

    def forward(self, x):
        # x = x.view(x.size(0), 1, 10, 10)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.deconv5(x)
        x = self.relu5(x)
        return x


class GeneratorSmall(nn.Module):
    def __init__(self, ngf):
        super(GeneratorSmall, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, 4*ngf, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4*ngf)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(4*ngf, 2*ngf, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2*ngf)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(2*ngf, ngf, 4, 2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)
        self.relu3 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(ngf, 1, 4, 2, padding=1, bias=False)
        self.relu4 = nn.Tanh()

    def forward(self, x):
        # x = x.view(x.size(0), 1, 10, 10)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        x = self.relu4(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
