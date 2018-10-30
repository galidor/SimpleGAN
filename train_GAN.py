import torch.utils.data
import torch.optim as optim
from PIL import Image
from utils import dataset_loader
import models
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
import torchvision

###################################
# TODO:
# try ReLU at the end of discriminator/generator EDIT: Don't
# investigate the source of 'borders' around numbers
# BatchNorm after activation
# normalize data
# parser
# organize document s.t. another designs can be added
####################################

batch_size = 64

writer = SummaryWriter('runs/test1')


def train(disc, gen, optim_disc, optim_gen, crit_disc, crit_gen, epoch, dataloader):
    loss_disc_sum = 0.0
    loss_gen_sum = 0.0
    disc.train()
    gen.train()
    print('Epoch {}'.format(epoch+1))
    for i, data in enumerate(dataloader, 0):

        true_imgs, _ = data
        true_imgs = true_imgs.cuda()
        pred_true = disc(true_imgs)
        loss_disc_true = crit_disc(pred_true, torch.ones_like(pred_true))

        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake_imgs = gen(noise)
        pred_fake = disc(fake_imgs)
        loss_disc_fake = crit_disc(pred_fake, torch.zeros_like(pred_fake))

        loss_disc = loss_disc_true + loss_disc_fake
        loss_disc_sum += loss_disc.item()
        disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()

        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake_imgs = gen(noise)
        pred_fake_gen = disc(fake_imgs)
        loss_gen = crit_gen(pred_fake_gen, torch.ones_like(pred_fake_gen))
        loss_gen_sum += loss_gen.item()
        disc.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # true_imgs, _ = data
        # true_imgs = true_imgs.cuda()
        # to_display = torchvision.utils.make_grid(true_imgs).cpu().numpy()
        # plt.imshow(np.transpose(to_display, (1, 2, 0)), interpolation='nearest')
        # plt.show()
        # true_imgs.requires_grad = False
        # gen.zero_grad()
        # noise = torch.randn(batch_size, 100).cuda()
        # noise = torch.reshape(noise, (batch_size, 100, 1, 1))
        # fake_imgs = gen(noise)
        # pred_fake_gen = disc(fake_imgs)
        # loss_gen = crit_gen(pred_fake_gen, torch.ones_like(pred_fake_gen))
        # loss_gen_sum += loss_gen.item()
        # loss_gen.backward()
        # optim_gen.step()
        #
        # disc.zero_grad()
        # pred_true = disc(true_imgs)
        # pred_fake = disc(fake_imgs.detach())
        # loss_disc = (crit_disc(pred_fake, torch.zeros_like(pred_fake)) +\
        #     crit_disc(pred_true, torch.ones_like(pred_true)))/2
        # loss_disc_sum += loss_disc.item()
        # loss_disc.backward()
        # optim_disc.step()
        if i % 100 == 0:
            print(i)
            im2show = torchvision.utils.make_grid(fake_imgs.cpu())
            trans = torchvision.transforms.ToPILImage()
            plt.imshow(trans(im2show))
            plt.show()
            writer.add_image('Images/test{}{}'.format(epoch+1, i), im2show)
    writer.add_scalar('disc_loss', loss_disc_sum, epoch + 1)
    writer.add_scalar('gen_loss', loss_gen_sum, epoch + 1)




mnist = dataset_loader.MNIST(root_dir='data/')
mnist_train, mnist_test = mnist.get_MNIST()

mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


discriminator = models.DiscriminatorSmall().cuda()
discriminator.initialize()
generator = models.GeneratorSmall().cuda()
generator.initialize()

# data = next(iter(mnist_train_loader))

# # img, label = data
# # print(img.shape)
# # img = img.cuda()
# # pred = discriminator(img)
# # print(pred.shape)
#
# noise = torch.randn(batch_size, 10).cuda()
# noise = torch.reshape(noise, (batch_size, 10, 1, 1))
# print(noise.shape)
# fake_img = generator(noise)
# print(fake_img.shape)
# print(fake_img.detach())
# fake_img = Image.fromarray(fake_img.detach().cpu().numpy().squeeze()).convert('L')

# plt.imshow(fake_img)
# plt.show()

optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_generator = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
crit_discriminator = torch.nn.BCELoss().cuda()
crit_generator = torch.nn.BCELoss().cuda()

epochs = 50
for epoch in range(epochs):
    train(discriminator, generator, optim_discriminator, optim_generator, crit_discriminator, crit_generator,
          epoch, mnist_train_loader)

writer.close()
