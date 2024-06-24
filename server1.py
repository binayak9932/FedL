from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict
import torch
import numpy as np
import os
import glob

from flwr.server import ServerApp, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Scalar, Parameters, FitIns
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from centralized import Net
from flwr.server.client_manager import ClientManager
import flwr as fl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(DEVICE)
num_clients=2



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
    
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")
            
            
            
            
        return aggregated_parameters, aggregated_metrics
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        # Initialize global model parameters.

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
    

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
    

# ... (keep all the imports and other definitions)

class IndefiniteTrainingStrategy(SaveModelStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_counter = 0
        self.training_rounds = 2  # Number of rounds to train when clients are available

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if self.round_counter == self.training_rounds:
            print(f"Completed {self.training_rounds} rounds of training. Resetting counter.")
            self.round_counter = 0
        
        return aggregated_parameters, aggregated_metrics

# Define strategy
strategy = IndefiniteTrainingStrategy(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients//2,
    min_available_clients=num_clients,
    evaluate_metrics_aggregation_fn=weighted_average
)

# Define config without specifying num_rounds
config = ServerConfig(round_timeout=3600)

# Flower ServerApp
app = ServerApp(config=config, strategy=strategy)

# Main execution
if __name__ == "__main__":
    from flwr.server import start_server
    import time

    print("Starting Federated Learning server. Press Ctrl+C to stop.")
    while True:
        try:
            start_server(
                server_address="0.0.0.0:8080",
                config=config,
                strategy=strategy,
            )
        except Exception as e:
            print(f"Server stopped: {e}")
            print("Restarting server in 10 seconds...")
            time.sleep(10)