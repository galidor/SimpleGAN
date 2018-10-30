import torchvision


class MNIST():
    def __init__(self, root_dir, download=False):
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.mnist_train = torchvision.datasets.MNIST(root_dir, train=True, transform=transforms,
                                                      target_transform=None, download=download)
        self.mnist_test = torchvision.datasets.MNIST(root_dir, train=False, transform=transforms,
                                                     target_transform=None, download=download)

    def get_MNIST(self):
        return self.mnist_train, self.mnist_train
