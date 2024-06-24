from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict
import torch
import numpy as np
import os
import glob
import time

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
num_clients=10



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_clients = num_clients

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
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        available_clients = client_manager.num_available()
         
        if available_clients < self.required_clients:
            print(f"Waiting for more clients. Available: {available_clients}, Required: {self.required_clients}")
            return []
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
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        # Initialize global model parameters.

        try:
            list_of_files = [fname for fname in glob.glob("./model_round_*")]
            latest_round_file = max(list_of_files, key=os.path.getctime)
            print("Loading pre-trained model from: ", latest_round_file)
            state_dict = torch.load(latest_round_file)
            net.load_state_dict(state_dict)
            state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
            parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
            initial_parameters = self.initial_parameters
            self.initial_parameters =parameters  
            return initial_parameters
        except:
            #initial_parameters = 
            return None
    

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
    

# Define strategy
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients // 2,
    min_available_clients=num_clients,
    evaluate_metrics_aggregation_fn=weighted_average
)


# Define config
config = ServerConfig(num_rounds=5, round_timeout=3600)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    def run_server():
        start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )

    while True:
        try:
            run_server()
        except Exception as e:
            print(f"Server stopped due to an error: {e}")
            print("Waiting before restarting server...")
            time.sleep(10)  # Wait for 10 seconds
            print("Restarting server...")
    
