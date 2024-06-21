import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
from flwr_datasets import FederatedDataset
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from centralized import Net, train, test

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda")
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS=2
BATCH_SIZE=32

def load_datasets(partition_id):
    fds = FederatedDataset(dataset='cifar10', partitioners={"train": NUM_CLIENTS}, shuffle=True, seed=42)

    def apply_transform(batch):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    partition = fds.load_partition(partition_id)
    partition = partition.train_test_split(train_size=.8, seed=42)
    partition = partition.with_transform(apply_transform)
   
    trainloader = DataLoader(partition["train"], shuffle=True, batch_size=BATCH_SIZE)
    valloader = DataLoader(partition["test"], batch_size=BATCH_SIZE)
        
    return trainloader, valloader

choices = list(map(int, range(NUM_CLIENTS)))

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=choices,
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)
partition_id = parser.parse_known_args()[0].partition_id

# Load model and data (simple CNN, CIFAR-10)

trainloader, valloader = load_datasets(partition_id=partition_id)
#net = Net().to(DEVICE)
#trainloader = trainloaders[0]
#valloader = valloaders[0]

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=5, verbose=True)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FlowerClient:
    """Create and return an instance of Flower `Client`."""
   
    return FlowerClient(net, trainloader, valloader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    net = Net().to(DEVICE)

    start_client(
        server_address="127.0.0.1:8080",    
        #client_fn=client_fn,
        client=FlowerClient(net, trainloader, valloader).to_client(),
        )
