from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class Rot_Dataset(Dataset):
    def __init__(self, root = './data', train = True):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        if train:
            self.imageset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        else:
            self.imageset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    def __len__(self):
        return len(self.imageset) * 4

    def __getitem__(self, idx):
        action = idx % 4
        img =  self.imageset.__getitem__(idx // 4)[0]
        img = transforms.functional.to_tensor(transforms.functional.rotate(img, 90 * action))
        img = transforms.functional.normalize(img, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return img, action