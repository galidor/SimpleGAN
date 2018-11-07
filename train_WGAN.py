import torch.utils.data
import torch.optim as optim
from PIL import Image
from utils import dataset_loader
import models
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
import torchvision
import argparse
import torch.nn as nn
# from torch.autograd import Variable
# from tensorflow.examples.tutorials.mnist import input_data

def parse():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('Required Arguments')
    optional = parser.add_argument_group('Optional Arguments')
    required.add_argument('--data_path', type=str, required=True, help='Specifies a path to directory containing'
                                                                       ' MNIST dataset, or where you wish to download')
    required.add_argument('--experiment_name', type=str, required=True, help='Name of your experiment for future'
                                                                             'reference in TensorBoard')

    optional.add_argument('--batch_size', type=int, default=64, help='Batch size')
    optional.add_argument('--optim_step', type=int, default=20, help='Number of epochs for learning rate decrease')
    optional.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    optional.add_argument('--normalize', action='store_true', help='Use if you want your data to be normalized')
    optional.add_argument('--download_dataset', action='store_true', help='Use it if you want to download MNIST to'
                                                                          ' DATA_PATH folder')
    optional.add_argument('--model_path', type=str, default='/', help='Path where your network will be'
                                                                                      ' stored')
    optional.add_argument('--test', action='store_true', help='Test mode only')
    optional.add_argument('--epochs', type=int, default=50, help='Total number of epochs')
    optional.add_argument('--ngf', type=int, default=128)
    optional.add_argument('--ndf', type=int, default=128)
    optional.add_argument('--clipping_threshold', type=float, default=0.01)
    optional.add_argument('--noise_length', type=int, default=100)

    args = parser.parse_args()
    return args


def train(disc, gen, optim_disc, optim_gen, crit_disc, crit_gen, epoch, dataloader, args, writer):
    loss_disc_sum = 0.0
    loss_gen_sum = 0.0
    disc.train()
    gen.train()
    iter_count = 0
    break_flag = False
    print('Epoch {}'.format(epoch+1))
    data = iter(dataloader)
    while(True):
        for _ in range(5):
            disc.zero_grad()
            gen.zero_grad()
            try:
                true_imgs, _ = next(data)
            except StopIteration:
                break_flag = True
                break
            true_imgs = true_imgs.cuda()
            noise = torch.randn(true_imgs.size(0), args.noise_length, 1, 1).cuda()
            fake_imgs = gen(noise)
            pred_true = disc(true_imgs)
            pred_fake = disc(fake_imgs)
            loss_disc = -(torch.mean(pred_true) - torch.mean(pred_fake))
            loss_disc_sum += loss_disc.item()/5.0
            loss_disc.backward()
            optim_disc.step()
            for p in disc.parameters():
                p.data.clamp_(-args.clipping_threshold, args.clipping_threshold)

        disc.zero_grad()
        gen.zero_grad()
        noise = torch.randn(true_imgs.size(0), args.noise_length, 1, 1).cuda()
        fake_imgs = gen(noise)
        pred_fake_gen = disc(fake_imgs)
        loss_gen = -torch.mean(pred_fake_gen)
        loss_gen_sum += loss_gen.item()
        loss_gen.backward()
        optim_gen.step()
        if break_flag:
            break

        iter_count += 1

        if iter_count % round(1000.0/float(args.batch_size)) == 0:
            im2show = torchvision.utils.make_grid(fake_imgs)
            writer.add_image('Images/generated_{}_{}'.format(epoch+1, i), im2show)
    writer.add_scalar('disc_loss', loss_disc_sum, epoch + 1)
    writer.add_scalar('gen_loss', loss_gen_sum, epoch + 1)


if __name__ == '__main__':

    opt = parse()

    transforms_list = [torchvision.transforms.Resize(28),
                       torchvision.transforms.ToTensor()]
    if opt.normalize:
        transforms_list.append(torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                std=(0.5, 0.5, 0.5)))
    transforms = torchvision.transforms.Compose(transforms_list)

    mnist = dataset_loader.MNIST(data_path=opt.data_path, transforms=transforms)
    mnist_train, mnist_test = mnist.get_MNIST()

    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=opt.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=opt.batch_size, shuffle=False)

    discriminator = models.DiscriminatorWasserstein(ndf=opt.ndf).cuda()
    # discriminator.initialize()
    generator = models.GeneratorWasserstein(ngf=opt.ngf, nz=opt.noise_length).cuda()
    # generator.initialize()
    discriminator.apply(models.weights_init)
    generator.apply(models.weights_init)

    optim_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=opt.learning_rate)
    optim_generator = torch.optim.RMSprop(generator.parameters(), lr=opt.learning_rate)
    crit_discriminator = torch.nn.BCELoss().cuda()
    crit_generator = torch.nn.BCELoss().cuda()

    writer = SummaryWriter('runs/{}'.format(opt.experiment_name))

    epochs = opt.epochs
    for epoch in range(epochs):
        train(discriminator, generator, optim_discriminator, optim_generator, crit_discriminator, crit_generator,
              epoch, mnist_train_loader, opt, writer)

    writer.close()
