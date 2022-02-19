import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# Download and store data.
def data_loader(batch_size = 10):
    """Download and store data in local directory.
       Dataset from: https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md
       return: tuple(train, test). Pytorch Tensors holding (images, target).
    """
    train = datasets.FashionMNIST(
        "FMINST data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test = datasets.FashionMNIST(
        "FMINST data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Load data into tensor, and shuffle
    train_set = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_set = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_set, test_set


# Usage example:
if __name__ == "__main__":
    train_set, test_set = data_loader()
    print(type(train_set))
    for data in train_set:
        images, target = data
        print(type(images), type(target))
        print(images.shape, target.shape)
        for i, image in enumerate(images):
            plt.title(target[i])
            plt.imshow(image.view(28, 28))
            #plt.show()
        break
