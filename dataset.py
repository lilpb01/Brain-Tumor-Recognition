from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # You can update this later based on dataset mean/std
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Testing", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes
