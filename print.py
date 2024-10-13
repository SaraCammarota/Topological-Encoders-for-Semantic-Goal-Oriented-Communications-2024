import hydra 
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def print_conf(config:DictConfig):
    print(config)

if __name__ == "__main__":

    print_conf()
