import time
from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict
import torch
import numpy as np
import os
import glob

from flwr.server import Server, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Scalar, Parameters, Metrics
from flwr.server.client_manager import ClientManager
import flwr as fl

from centralized import Net  # Make sure this import is correct

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(DEVICE)
NUM_CLIENTS = 2
REQUIRED_CLIENTS = 2
TRAINING_ROUNDS = 2

class IndefiniteTrainingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_counter = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        self.round_counter += 1
        if self.round_counter == TRAINING_ROUNDS:
            print(f"Completed {TRAINING_ROUNDS} rounds of training.")
            if aggregated_metrics:
                print(f"Updated accuracy: {aggregated_metrics.get('accuracy', 'N/A')}")
            self.round_counter = 0

        return aggregated_parameters, aggregated_metrics
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        try:
            list_of_files = glob.glob("./model_round_*.pth")
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                print(f"Loading pre-trained model from: {latest_file}")
                state_dict = torch.load(latest_file)
                net.load_state_dict(state_dict)
                state_dict_ndarrays = [v.cpu().numpy() for v in state_dict.values()]
                return fl.common.ndarrays_to_parameters(state_dict_ndarrays)
            else:
                print("No saved models found. Initializing with random weights.")
        except Exception as e:
            print(f"Error loading saved model: {e}")
            print("Initializing with random weights.")

        state_dict = net.state_dict()
        state_dict_ndarrays = [v.cpu().numpy() for v in state_dict.values()]
        return fl.common.ndarrays_to_parameters(state_dict_ndarrays)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = IndefiniteTrainingStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=REQUIRED_CLIENTS,
    min_evaluate_clients=REQUIRED_CLIENTS,
    min_available_clients=REQUIRED_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average
)

class CustomServer(Server):
    def fit(self, num_rounds: int, timeout: Optional[float]) -> None:
        for server_round in range(1, num_rounds + 1):
            print(f"Starting round {server_round}")
            print("Waiting for clients...")
            while self.client_manager().num_available() < REQUIRED_CLIENTS:
                time.sleep(5)
            print(f"Starting training with {self.client_manager().num_available()} clients")
            res = self.fit_round(server_round, timeout)
            if res is None:
                break

if __name__ == "__main__":
    print("Starting Federated Learning server. Press Ctrl+C to stop.")
    while True:
        try:
            server = CustomServer(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
            fl.server.start_server(
                server_address="0.0.0.0:8080",  # Using IPv4
                server=server,
                config=ServerConfig(num_rounds=TRAINING_ROUNDS),
            )
        except Exception as e:
            print(f"Server stopped: {e}")
            print("Restarting server in 10 seconds...")
            time.sleep(10)