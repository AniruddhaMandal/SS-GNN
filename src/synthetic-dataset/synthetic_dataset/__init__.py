from .triangles import ParityTriangleGraphDataset
from .cliques import K4ParityDataset
from .clique_detection import CliqueDetectionDataset, MultiCliqueDetectionDataset
from .clique_detection_controlled import DensityControlledCliqueDetectionDataset
from .sparse_clique_detection import SparseCliqueDetectionDataset
from .csl import CSLDataset
import os, json, hashlib, torch
from pathlib import Path


# synthetic_dataset/factory.py
import os
import json
import hashlib
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Tuple, List

import torch
from torch_geometric.data import InMemoryDataset, Data


# ---------- helpers ----------
def _stable_hash_from_dict(d: Dict[str, Any]) -> str:
    """Create a stable short hash for a JSON-serializable dict of parameters."""
    # Ensure deterministic serialization
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _atomic_save(obj: Any, path: str) -> None:
    """Write the object to a temporary file and atomically rename into place."""
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp_cache_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ---------- InMemory wrapper ----------
class ListAsInMemoryDataset(InMemoryDataset):
    """
    Wrap a list of torch_geometric.data.Data into an InMemoryDataset so it loads
    naturally into PyG DataLoader and utilities.
    """
    def __init__(self, data_list: List[Data], transform=None):
        super().__init__(root=None, transform=transform)
        # store raw_list (not part of InMemoryDataset pattern necessarily),
        # but we also set self.data, self.slices for compatibility using torch_geometric helper
        self._list = data_list
        # Build data and slices in the standard InMemoryDataset way:
        data, slices = self.collate(data_list)
        self.data, self.slices = data, slices

    def len(self) -> int:  # for older PyG naming compatibility
        return len(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def get(self, idx: int) -> Data:
        return self._list[idx]


# ---------- Factory dataclass ----------
@dataclass
class SyntheticGraphData:
    """
    Factory that constructs or loads cached synthetic datasets by name.

    Usage:
        factory = DatasetFactory(cache_dir="~/.cache/SYNTHETIC")
        ds = factory.get("triangle", num_graphs=100, node_range=(8,12), p=0.3, desired_parity=1)

    The call will:
      - check cache_dir for a cached file matching ("triangle" + params) hash
      - if found, load and return the dataset (InMemoryDataset wrapper)
      - otherwise, construct the dataset via the mapped constructor, save it to cache, and return it
    """

    cache_dir: str = field(default_factory=lambda: os.path.join(os.path.expanduser("~"), ".cache", "SYNTHETIC-DATA"))
    use_inmemory_wrapper: bool = True
    # mapping name -> callable that constructs the dataset. You can append more mappings.
    registry: Dict[str, Callable[..., Any]] = field(default_factory=lambda: {
        "Triangle-Parity": ParityTriangleGraphDataset,
        "K4": K4ParityDataset,
        "Clique-Detection": CliqueDetectionDataset,
        "Multi-Clique-Detection": MultiCliqueDetectionDataset,
        "Clique-Detection-Controlled": DensityControlledCliqueDetectionDataset,
        "Sparse-Clique-Detection": SparseCliqueDetectionDataset,
        "CSL": CSLDataset,
    })

    def _make_cache_paths(self, name: str, params: Dict[str, Any]) -> Tuple[str, str]:
        """
        Return (cache_file_path, meta_file_path).
        cache file stores dataset (torch.save)
        meta file stores json metadata (human-readable)
        """
        safe_name = name.replace(os.sep, "_")
        short_hash = _stable_hash_from_dict(params)
        base = f"{safe_name}_{short_hash}"
        cache_file = os.path.join(self.cache_dir, f"{base}.pt")
        meta_file = os.path.join(self.cache_dir, f"{base}.meta.json")
        return cache_file, meta_file

    def exists_in_cache(self, name: str, params: Dict[str, Any]) -> bool:
        cache_file, _ = self._make_cache_paths(name, params)
        return os.path.exists(cache_file)

    def clear_cache(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Remove cached files. If name is None -> clear entire cache_dir.
        If params is None but name provided -> clear all cached files for that name prefix.
        """
        if name is None:
            # remove everything in cache_dir
            if os.path.exists(self.cache_dir):
                for fn in os.listdir(self.cache_dir):
                    try:
                        os.remove(os.path.join(self.cache_dir, fn))
                    except Exception:
                        pass
            return

        # otherwise remove specific files
        if params is None:
            # remove files that start with name_
            prefix = f"{name}_"
            for fn in os.listdir(self.cache_dir) if os.path.exists(self.cache_dir) else []:
                if fn.startswith(prefix):
                    try:
                        os.remove(os.path.join(self.cache_dir, fn))
                    except Exception:
                        pass
            return

        cache_file, meta_file = self._make_cache_paths(name, params)
        for path in (cache_file, meta_file):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def register(self, name: str, constructor: Callable[..., Any], overwrite: bool = False) -> None:
        """Register a new dataset constructor under `name`."""
        if name in self.registry and not overwrite:
            raise ValueError(f"Dataset name '{name}' already registered. Use overwrite=True to replace.")
        self.registry[name] = constructor

    def _serialize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make params JSON-serializable for hashing and metadata file.
        Convert non-serializable items (e.g., devices, callables) to strings.
        """
        def _clean(v):
            if isinstance(v, (str, bool, int, float, type(None))):
                return v
            if isinstance(v, dict):
                return {k: _clean(v[k]) for k in sorted(v)}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return str(v)
        return _clean(params)

    def get(self, name: str, cache: bool = True, **constructor_kwargs) -> InMemoryDataset:
        """
        Get dataset by name.

        Parameters:
            name: registered dataset name ("triangle", "k4", or custom-registered)
            cache: whether to use cache (if True, attempt to load, otherwise always recreate)
            **constructor_kwargs: forwarded to the dataset constructor

        Returns:
            A torch_geometric InMemoryDataset (ListAsInMemoryDataset) containing Data objects.
        """
        if name not in self.registry:
            raise KeyError(f"No dataset registered under name '{name}'. Registered names: {list(self.registry.keys())}")

        constructor = self.registry[name]
        # Include all relevant parameters in cache key, not just num_graphs and node_range
        # This ensures different parameter combinations are cached separately
        specification = {k: v for k, v in constructor_kwargs.items()
                        if k not in ['transform', 'pre_transform', 'seed', 'store_on_device', 'device']}
        params_clean = self._serialize_params(specification)
        cache_file, meta_file = self._make_cache_paths(name, params_clean)

        if cache and os.path.exists(cache_file):
            # load cached dataset
            loaded = torch.load(cache_file, weights_only=False)
            # loaded is expected to be a list of Data or an InMemoryDataset; normalize
            if isinstance(loaded, InMemoryDataset):
                return loaded
            if isinstance(loaded, list):
                if self.use_inmemory_wrapper:
                    return ListAsInMemoryDataset(loaded)
                else:
                    # wrap as minimal object that has __len__ and __getitem__
                    return loaded  # type: ignore
            # unknown type -> return as-is
            return loaded  # type: ignore

        # need to create
        # create an instance of the generator. Many of your existing classes are Dataset subclasses,
        # but they might precompute graphs in __init__ and return Data objects via __getitem__.
        # We'll attempt to detect and convert into a list of Data objects.
        instance = constructor(**constructor_kwargs)

        # If constructor returns a torch.utils.data.Dataset (like your previous classes),
        # collect all Data objects into a list
        data_list: List[Data]
        if hasattr(instance, "__len__") and hasattr(instance, "__getitem__"):
            # collect
            data_list = [instance[i] for i in range(len(instance))]
        elif isinstance(instance, list):
            data_list = instance
        else:
            # Unexpected return type; try to treat it as an iterable
            try:
                data_list = list(instance)
            except Exception as exc:
                raise RuntimeError(f"Cannot convert constructed object to list of Data: {exc}")

        # Save to cache if requested
        if cache:
            # We store the plain list of Data objects; on load we wrap back into InMemoryDataset.
            _atomic_save(data_list, cache_file)
            # Save metadata (human readable)
            meta = {
                "name": name,
                "params": params_clean,
                "created_by": constructor.__name__ if hasattr(constructor, "__name__") else str(constructor),
                "num_items": len(data_list),
            }
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, sort_keys=True)
            except Exception:
                pass

        if self.use_inmemory_wrapper:
            return ListAsInMemoryDataset(data_list)
        return data_list  # type: ignore
