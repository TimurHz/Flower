import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn

@hydra.main(config_path="conf",config_name="base", version_base=None)


def main(cfg: DictConfig):
    
    #print config 
    print(OmegaConf.to_yaml(cfg))
    
    #Datensatz vorbereiten
    trainloaders, validationloaders, testloaders = prepare_dataset(cfg.num_clients, cfg.batch_size)
    
    print(len(trainloaders), len(trainloaders[0].dataset))

    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)
    
if __name__ == "__main__":
    main()