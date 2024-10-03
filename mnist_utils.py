import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch_geometric.data import Data
import torch

# Custom Dataset for MNIST Graphs
class MNISTGraphDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        return self.image_to_graph(image, label)

    def image_to_graph(self, image, label):
        # Flatten the image into a vector (N x 1) where N = num of pixels (28x28 = 784)
        x = image.view(-1, 1)  # Node features, shape (N x 1), where N = 784
        
        # Create the edges (4-neighbor connectivity)
        N = image.shape[1]  # N = 28, since images are 28x28
        edge_index = []
        
        # Define 4-neighbor connectivity (up, down, left, right)
        for i in range(N):
            for j in range(N):
                node_idx = i * N + j  # Flattened pixel index
                if i > 0:  # Up
                    edge_index.append([node_idx, node_idx - N])
                if i < N - 1:  # Down
                    edge_index.append([node_idx, node_idx + N])
                if j > 0:  # Left
                    edge_index.append([node_idx, node_idx - 1])
                if j < N - 1:  # Right
                    edge_index.append([node_idx, node_idx + 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create a PyG Data object for each image
        data = Data(x=x, edge_index=edge_index, y=label)

        data.batch_0 = data.batch  # Assign current batch to batch_0
        del data.batch
        return data


# PyTorch Lightning DataModule
class MNISTGraphDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Download the dataset here (if not already downloaded)
        datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage=None):
        # Load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        
        # Convert to graph datasets
        self.train_dataset = MNISTGraphDataset(mnist_train)
        self.test_dataset = MNISTGraphDataset(mnist_test)

        # Split training into train/val sets
        train_len = int(0.8 * len(self.train_dataset))
        val_len = len(self.train_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
