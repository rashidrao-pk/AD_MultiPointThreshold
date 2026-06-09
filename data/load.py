from .mvtec_loading import get_dataloaders_mvtec
from .cobots_loading import get_dataloaders_cobots
from .corridor_loading import get_dataloaders_corridor

# This is a wrapper to load data based on name,
# in case we want to add more datasets in the future, we can just add more options here, with their specific loading functions.
# it will return train_loader, test_loader, train_dataset, test_dataset



def load_data(cfg):
    if cfg.data.name == "MVTec":
        return get_dataloaders_mvtec(cfg.data)
    
    if cfg.data.name == "Cobots_Synthetic":
        return get_dataloaders_cobots(cfg.data)
    
    if cfg.data.name == "Robotics_Hazards":
        return get_dataloaders_corridor(cfg.data)

    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")
    