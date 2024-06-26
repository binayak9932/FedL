import argparse
import warnings
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
import flwr as fl
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

BATCH_SIZE = 32

# Get port
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--port",
    type=int,
    default=8081,
    help="Port to use for this client"
)
args = parser.parse_args()
port = args.port

class XrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_path):
                images.append((os.path.join(class_path, img_name), self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {"img": image, "label": label}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)  
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train(net, trainloader, epochs: int, verbose=False):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in tqdm(trainloader, "Training"):
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total * 100
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            
            
def test(net, testloader):
    criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE).float().unsqueeze(1)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy



def load_datasets(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    trainset = XrayDataset(train_dir, transform=transform)
    testset = XrayDataset(test_dir, transform=transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, testloader

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
        train(self.net, self.trainloader, epochs=10, verbose=True)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    """Create and return an instance of Flower `Client`."""
    net = Net().to(DEVICE)
    trainloader, valloader = load_datasets(data_dir="./xray_dataset_covid19")  # Update this path
    return FlowerClient(net, trainloader, valloader).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Main execution
if __name__ == "__main__":
    from flwr.client import start_client

    net = Net().to(DEVICE)
    trainloader, valloader = load_datasets(data_dir="./xray_dataset_covid19")  # Update this path

    start_client(
        server_address=f"localhost:8090",
        client=FlowerClient(net, trainloader, valloader).to_client(),
    )