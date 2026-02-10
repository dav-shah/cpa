from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter, AnnDataLoader
from scvi.model._utils import parse_use_gpu_arg
from scipy import sparse
import torch
import numpy as np


class SparseToDenseDataLoader:
    """
    Wrapper that converts sparse matrices to dense tensors during batch iteration.
    This enables memory-efficient sparse storage while maintaining model compatibility.
    
    Used for Optimization 1.1: Sparse Perturbation Storage
    """
    def __init__(self, dataloader, sparse_keys=None):
        self.dataloader = dataloader
        self.sparse_keys = sparse_keys or ['perts', 'perts_doses']
    
    def __iter__(self):
        for batch in self.dataloader:
            # Convert sparse matrices to dense tensors
            for key in self.sparse_keys:
                if key in batch:
                    if sparse.issparse(batch[key]):
                        # Convert scipy sparse matrix to dense array, then to tensor
                        dense_array = batch[key].toarray()
                        batch[key] = torch.from_numpy(dense_array)
                    elif isinstance(batch[key], np.ndarray):
                        # Already dense numpy array, just convert to tensor
                        if not isinstance(batch[key], torch.Tensor):
                            batch[key] = torch.from_numpy(batch[key])
            yield batch
    
    def __len__(self):
        return len(self.dataloader)


class SparseAwareDataSplitter(DataSplitter):
    """
    DataSplitter that wraps dataloaders to convert sparse matrices to dense tensors.
    
    Used for Optimization 1.1: Sparse Perturbation Storage
    """
    def __init__(self, *args, sparse_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_keys = sparse_keys or ['perts', 'perts_doses']
    
    def train_dataloader(self):
        loader = super().train_dataloader()
        return SparseToDenseDataLoader(loader, self.sparse_keys) if loader else None
    
    def val_dataloader(self):
        loader = super().val_dataloader()
        return SparseToDenseDataLoader(loader, self.sparse_keys) if loader else None
    
    def test_dataloader(self):
        loader = super().test_dataloader()
        return SparseToDenseDataLoader(loader, self.sparse_keys) if loader else None


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata_manager: AnnDataManager,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            sparse_keys=None,
            **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices
        # Keys for sparse-to-dense conversion (Optimization 1.1)
        self.sparse_keys = sparse_keys or ['perts', 'perts_doses']

    def setup(self, stage: Optional[str] = None):
        accelerator, _, self.device = parse_use_gpu_arg(
            self.use_gpu, return_device=True
        )
        self.pin_memory = (
            True
            if (settings.dl_pin_memory_gpu_training and accelerator == "gpu")
            else False
        )

    def train_dataloader(self):
        if len(self.train_idx) > 0:
            loader = AnnDataLoader(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
            # Wrap to convert sparse to dense during iteration (Optimization 1.1)
            return SparseToDenseDataLoader(loader, self.sparse_keys)
        else:
            pass

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            # if len(self.valid_indices < 4096):
            #     data_loader_kwargs.update({'batch_size': len(self.valid_indices)})
            # else:
            #     data_loader_kwargs.update({'batch_size': 2048})
            loader = AnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
            # Wrap to convert sparse to dense during iteration (Optimization 1.1)
            return SparseToDenseDataLoader(loader, self.sparse_keys)
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            loader = AnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
            # Wrap to convert sparse to dense during iteration (Optimization 1.1)
            return SparseToDenseDataLoader(loader, self.sparse_keys)
        else:
            pass
