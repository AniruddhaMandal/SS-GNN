from .registry import register_dataset
from . import ExperimentConfig
from .utils.split_and_loader import build_dataloaders_from_dataset

@register_dataset('K4')
@register_dataset('Triangle-Parity')
@register_dataset('Clique-Detection')
@register_dataset('Multi-Clique-Detection')
@register_dataset('Clique-Detection-Controlled')
@register_dataset('Sparse-Clique-Detection')
@register_dataset('CSL')
def build_synthetic(cfg: ExperimentConfig):
    from synthetic_dataset import SyntheticGraphData
    from torch_geometric.transforms import ToUndirected, Compose
    from .utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed, SetNodeFeaturesOnes, AddLaplacianPE

    f_type = cfg.model_config.kwargs.get('node_feature_type')
    max_degree = cfg.model_config.kwargs.get('max_degree')
    lap_pe_dim = cfg.model_config.kwargs.get('lap_pe_dim', 8)  # Default 8 eigenvectors

    assert  f_type is not None,  \
        "for data with no feature type requires `node_feature_type` in model keywords."

    if f_type == "all_one":
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([SetNodeFeaturesOnes(dim=node_dim,cat=False)])
    elif f_type == "lap_pe":
        # Laplacian Positional Encoding (recommended for CSL)
        transforms = Compose([AddLaplacianPE(k=lap_pe_dim, cat=False)])
    elif f_type == "all_one_with_lap_pe":
        # All-one features concatenated with Laplacian PE
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([
            SetNodeFeaturesOnes(dim=node_dim, cat=False),
            AddLaplacianPE(k=lap_pe_dim, cat=True)
        ])
    elif f_type == "one_hot_degree":
        assert max_degree is not None,  \
            "`max_degree` in model keywords. "
        transforms = Compose([
            ToUndirected(),
            ClipOneHotDegree(max_degree=max_degree,cat=False)
            ])
    elif f_type == "degree_embed":
        assert max_degree is not None,  \
            "`max_degree` in model keywords. "
        node_dim = cfg.model_config.node_feature_dim
        transforms = Compose([
            ToUndirected(),
            ClipDegreeEmbed(max_degree=max_degree,embed_dim=node_dim,cat=False)
            ])
    else:
        raise ValueError(f"Unknown `node_feature_type`({f_type})")

    syn_data = SyntheticGraphData(cache_dir='./data/SYNTHETIC-DATA')

    # Different parameters for different synthetic datasets
    if cfg.dataset_name == 'CSL':
        dataset = syn_data.get(cfg.dataset_name, cache=True, transform=transforms)
        return build_dataloaders_from_dataset(dataset, cfg)
    if cfg.dataset_name == 'Clique-Detection':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(20, 40),
                               p_no_clique=0.04,
                               p_with_clique=0.08,
                               transform=transforms)
    elif cfg.dataset_name == 'Clique-Detection-Controlled':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(20, 30),
                               p_no_clique=0.08,
                               p_with_clique=0.06,
                               transform=transforms)
    elif cfg.dataset_name == 'Multi-Clique-Detection':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(25, 45),
                               p_base=0.08,
                               transform=transforms)
    elif cfg.dataset_name == 'Sparse-Clique-Detection':
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               k=4,
                               node_range=(30, 50),
                               p_base=0.015,
                               transform=transforms)
    elif cfg.dataset_name == 'Triangle-Parity':
        # Original triangle parity dataset
        data_even = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=1000,
                               node_range=(20,40),
                               desired_parity=0,
                               p=0.1,
                               transform=transforms)
        data_odd = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=1000,
                               node_range=(20,40),
                               desired_parity=1,
                               p=0.1,
                               transform=transforms)
        dataset = data_even + data_odd
    else:
        # Default for K4 and other datasets
        dataset = syn_data.get(cfg.dataset_name,
                               cache=True,
                               num_graphs=2000,
                               node_range=(50,60),
                               desired_parity=[0,1],
                               p=0.051,
                               transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)
    

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
@register_dataset('PTC_MR')
@register_dataset('AIDS')
def build_tudata(cfg: ExperimentConfig):
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import ToUndirected, Compose
    from .utils.data_transform import ClipOneHotDegree, ClipDegreeEmbed
    transforms = Compose([ToUndirected()])
    dataset = TUDataset(root="data/TUDataset",name=cfg.dataset_name,transform=transforms)
    needs_x = (getattr(dataset[0],'x', None) is None) or (dataset.num_node_features == 0)
    if needs_x:
        assert hasattr(cfg.model_config, 'node_feature_type') # for data with no feature type requires `node_feature_type`
        assert hasattr(cfg.model_config, 'max_degree') # and `max_degree`
        max_degree = cfg.model_config.max_degree
        if cfg.model_config.node_feature_type == "one_hot_degree":
            transforms = Compose([ToUndirected(),ClipOneHotDegree(max_degree=max_degree,
                                                                  cat=False)])
        elif cfg.model_config.node_feature_type == "degree_embed":
            node_dim = cfg.model_config.node_feature_dim
            transforms = Compose([ToUndirected(),ClipDegreeEmbed(max_degree=max_degree,
                                                                 embed_dim=node_dim,
                                                                 cat=False)])
        else:
            raise ValueError(f"Unknown `node_feature_type`({cfg.model_config.node_feature_type})")
        dataset = TUDataset(root="data/TUDataset",name=cfg.dataset_name,transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)

@register_dataset('ZINC')
def build_zinc(cfg: ExperimentConfig):
    from torch_geometric.datasets import ZINC
    from .encoder import AtomBondEncoder
    from torch_geometric.transforms import Compose, ToUndirected
    transforms = Compose([AtomBondEncoder(cfg.model_config.node_feature_dim,
                        cfg.model_config.edge_feature_dim, requirs_grad=False), ToUndirected()])
    train_dataset = ZINC(root='./data/ZINC', subset=True, split='train',
                        pre_transform=transforms)
    test_dataset = ZINC(root='./data/ZINC', subset=True, split='test',
                        pre_transform=transforms)
    val_dataset = ZINC(root='./data/ZINC', subset=True, split='val',
                        pre_transform=transforms)
    dataset = (train_dataset, test_dataset, val_dataset)
    return build_dataloaders_from_dataset(dataset, cfg)

@register_dataset('QM9')
def build_qm9(cfg: ExperimentConfig):
    from torch_geometric.datasets import QM9
    from sklearn.model_selection import train_test_split
    from .encoder import FilterTarget
    from .encoder import NormaliseTarget


    if cfg.task == 'Single-Target-Regression':
        if cfg.train.dataloader_kwargs.get('target_idx', None) is None:
            raise NameError(f'set target_idx in config.')
        target_idx = int(cfg.train.dataloader_kwargs.get('target_idx'))
        transforms = FilterTarget(target_idx)
    dataset = QM9('./data/QM9',transform=transforms)
    if cfg.task == "All-Target-Regression":
        transforms = None

    return build_dataloaders_from_dataset(dataset, cfg)

@register_dataset('PPI-BP')
@register_dataset('HPO-METAB')
@register_dataset('HPO-NEURO')
@register_dataset('EM-USER')
def build_subgnn(cfg: ExperimentConfig):
    """
    Build SubGNN datasets (Alsentzer et al., NeurIPS 2020)

    Datasets:
    - PPI-BP: Protein-protein interaction - Biological process (6 classes)
    - HPO-METAB: Metabolic disorder classification (6 classes)
    - HPO-NEURO: Neurological disorder classification (10 classes, multi-label)
    - EM-USER: Endomondo user gender classification (2 classes)
    """
    from .subgnn_dataset import SubGNNDataset
    from torch_geometric.transforms import Compose
    from .utils.data_transform import AddLaplacianPE, SetNodeFeaturesOnes

    # Map dataset name to internal format
    dataset_name_map = {
        'PPI-BP': 'ppi_bp',
        'HPO-METAB': 'hpo_metab',
        'HPO-NEURO': 'hpo_neuro',
        'EM-USER': 'em_user'
    }

    internal_name = dataset_name_map[cfg.dataset_name]

    # Optional transforms
    transforms = None
    if hasattr(cfg.model_config, 'kwargs'):
        f_type = cfg.model_config.kwargs.get('node_feature_type', None)
        lap_pe_dim = cfg.model_config.kwargs.get('lap_pe_dim', 8)

        if f_type == "lap_pe":
            transforms = Compose([AddLaplacianPE(k=lap_pe_dim, cat=False)])
        elif f_type == "all_one_with_lap_pe":
            node_dim = cfg.model_config.node_feature_dim
            transforms = Compose([
                SetNodeFeaturesOnes(dim=node_dim, cat=False),
                AddLaplacianPE(k=lap_pe_dim, cat=True)
            ])

    # Load dataset
    dataset = SubGNNDataset(
        root='./data/SubGNN',
        name=internal_name,
        transform=transforms
    )

    # Get splits
    splits = dataset.get_idx_split()
    train_dataset = dataset[splits['train']]
    val_dataset = dataset[splits['valid']]
    test_dataset = dataset[splits['test']]

    # Return as tuple for build_dataloaders_from_dataset
    return build_dataloaders_from_dataset((train_dataset, test_dataset, val_dataset), cfg)

@register_dataset('ogbg-molhiv')
@register_dataset('molhiv')
def build_molhiv(cfg: ExperimentConfig):
    """
    Build OGB-MolHIV dataset.

    Uses RDKit for SMILES-to-graph conversion with OGB-style features.
    Scaffold splitting (80/10/10) is applied.

    Task: Binary classification (HIV activity prediction)
    Metric: ROC-AUC
    """
    from .dataset_loaders.molhiv import MolHIVDataset
    from .encoder import OGBAtomEncoder, OGBBondEncoder
    from torch_geometric.transforms import Compose

    emb_dim = cfg.model_config.node_feature_dim
    edge_emb_dim = getattr(cfg.model_config, 'edge_feature_dim', None) or emb_dim

    transforms = Compose([
        OGBAtomEncoder(emb_dim=emb_dim),
        OGBBondEncoder(emb_dim=edge_emb_dim),
    ])

    dataset = MolHIVDataset(
        root='./data/OGB/molhiv',
        transform=transforms
    )

    # Dataset has get_idx_split() - will be used automatically by build_dataloaders_from_dataset
    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('BBBP')
def build_bbbp(cfg: ExperimentConfig):
    """
    Build BBBP (Blood-Brain Barrier Penetration) dataset from MoleculeNet.

    Task: Binary classification
    Size: ~2039 molecules
    Node features: 9 (atom features from RDKit)
    """
    from torch_geometric.datasets import MoleculeNet
    from torch_geometric.transforms import ToUndirected, Compose

    transforms = Compose([ToUndirected()])
    dataset = MoleculeNet(root='data/MoleculeNet', name='BBBP', transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('Tox21')
def build_tox21(cfg: ExperimentConfig):
    """
    Build Tox21 dataset from MoleculeNet.

    Task: Multi-label binary classification (12 toxicity assays)
    Size: ~7831 molecules
    Node features: 9 (atom features from RDKit)
    Note: Labels contain NaN values for missing assays
    """
    from torch_geometric.datasets import MoleculeNet
    from torch_geometric.transforms import ToUndirected, Compose

    transforms = Compose([ToUndirected()])
    dataset = MoleculeNet(root='data/MoleculeNet', name='Tox21', transform=transforms)

    return build_dataloaders_from_dataset(dataset, cfg)
