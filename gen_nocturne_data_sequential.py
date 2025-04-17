import time
import hydra
from omegaconf import DictConfig
from gen_data import NocturneDatasetGenerator
from pprint import pprint

@hydra.main(config_name="nocturne_cfg.yaml", config_path="./", version_base="1.3")
def main(cfg: DictConfig):
    dataset = NocturneDatasetGenerator(config=cfg.nocturne)
    
    # Start time    
    start_time = time.time()
    if cfg.nocturne.dataset_mode == "graph":
        dataset.gen_graph_dataset()
    elif cfg.nocturne.dataset_mode == "image":
        dataset.gen_image_dataset(parallel=False)

    # End time
    end_time = time.time()
    # Calculate and print total running time
    total_time_sequential = end_time - start_time
    print(f"Total running time (sequential): {total_time_sequential:.2f} seconds")

if __name__ == "__main__":
    main()
