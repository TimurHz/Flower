import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
import flwr as fl
from server import get_on_fit_config, get_eval_fn

@hydra.main(config_path="conf",config_name="base", version_base=None)


def main(cfg: DictConfig):
    
    #print config 
    print(OmegaConf.to_yaml(cfg))
    
    #Datensatz vorbereiten
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
    
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)


    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ), 
        evaluate_fn=get_eval_fn(cfg.num_classes, testloader),
    ) 
    

    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy
    )


if __name__ == "__main__":
    main()