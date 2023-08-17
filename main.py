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


    startegy = fl.server.strategy.FedAvg(fraction_fit=0.000001, 
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.000001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_eval_fn(cfg.num_classes, testloader))
    
if __name__ == "__main__":
    main()