from .registry import register_dataset
from . import ExperimentConfig
from .utils.split_and_loader import build_dataloaders_from_dataset

@register_dataset('PascalVOC-SP')
@register_dataset('COCO-SP')
@register_dataset('PCQM-Contact')
@register_dataset('Peptides-func')
@register_dataset('Peptides-struct')
def build_lrgb(cfg: ExperimentConfig):
    from torch_geometric.datasets import LRGBDataset
    train_data = LRGBDataset("data/LRGB",cfg.dataset_name, "train")
    test_data = LRGBDataset("data/LRGB", cfg.dataset_name, "test")
    val_data = LRGBDataset("data/LRGB", cfg.dataset_name, "val")
    dataset = (train_data, test_data, val_data)
    return build_dataloaders_from_dataset(dataset, cfg)

@register_dataset('MUTAG')
@register_dataset('ENZYMES')
@register_dataset('PROTEINS')
@register_dataset('COLLAB')
@register_dataset('IMDB-BINARY')
@register_dataset('REDDIT-BINARY')
def build_tudata(cfg: ExperimentConfig):
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import ToUndirected, Compose
    from .utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed
    transforms = Compose([
        ToUndirected()
    ])
    dataset = TUDataset(root="data/TUDataset", 
                        name=cfg.dataset_name, 
                        transform=transforms)
    
    needs_x = (getattr(dataset[0],'x', None) is None) or (dataset.num_node_features == 0)
    if needs_x:
        assert hasattr(cfg.model_config, 'node_feature_type') # for data with no feature type requires `node_feature_type`
        assert hasattr(cfg.model_config, 'max_degree') # and `max_degree`
        if cfg.model_config.node_feature_type == "one_hot_degree":
            transforms = Compose([
                ToUndirected(),
                ClipOneHotDegree(max_degree=cfg.model_config.max_degree, cat=False)
            ])
        if cfg.model_config.node_feature_type == "degree_embed":
            transforms = Compose([
                ToUndirected(),
                ClipDegreeEmbed(max_degree=cfg.model_config.max_degree, 
                                embed_dim=cfg.model_config.node_feature_dim, 
                                cat=False)
            ])
        else:
            raise ValueError(f"Unknown `node_feature_type`({cfg.model_config.node_feature_type})")
        dataset = TUDataset(root="data/TUDataset", 
                            name=cfg.dataset_name, 
                            transform=transforms, 
                            )

    return build_dataloaders_from_dataset(dataset, cfg)
