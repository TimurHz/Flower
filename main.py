import pickle

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from client import generate_client_fn
import flwr as fl
from server import get_on_fit_config, get_eval_fn, globaleval
from dataset import prepare_dataset
from pathlib import Path

from model import Net, test
import torch
from collections import OrderedDict




@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # print config
    print(OmegaConf.to_yaml(cfg))

    # Datensatz vorbereiten
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.000001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_eval_fn(cfg.num_classes, testloader))
    
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        #client_resources={
        #    'num_cpus': 2,      # Wie viele cpus sollte ein client haben
        #    'num_gpus': 0     # Wie viel des vrams die clients zur verfügung haben. Für =1 kann nur 1 client gleichz. laufen
        #}
    )

    print("------------------------")
    
    


    save_path = HydraConfig.get().runtime.output_dir
    result_path = Path(save_path) / 'results.pkl'

    results = {'history': history}
    with open(str(result_path), 'wb') as h:
        pickle .dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()