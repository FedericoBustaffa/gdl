from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def mnist(train_size, test_size, batch_size):
    training_data = MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = MNIST(
        root="datasets",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_subset, _ = random_split(
        training_data,
        [train_size, len(training_data) - train_size],
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    test_subset, _ = random_split(test_data, [test_size, len(test_data) - test_size])
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
