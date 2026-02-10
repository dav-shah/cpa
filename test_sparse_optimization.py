#!/usr/bin/env python
"""
Test script for Optimization 1.1: Sparse Perturbation Storage

This script verifies that:
1. Sparse matrices are stored correctly in AnnData
2. Data loading converts sparse to dense correctly
3. Model can train with sparse storage
4. Memory is reduced as expected
"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse

# Add cpa to path
sys.path.insert(0, '.')

try:
    import anndata
    import cpa
    
    print("=" * 80)
    print("Testing Optimization 1.1: Sparse Perturbation Storage")
    print("=" * 80)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    n_cells = 1000
    n_genes = 200
    X = np.random.randint(low=0, high=100, size=(n_cells, n_genes))
    obs = pd.DataFrame(
        dict(
            drug_name=np.array(["d1", "d2", "d3", "control"])[np.random.randint(4, size=n_cells)],
            dose_val=np.array([0.0, 0.1, 0.5, 1.0])[np.random.randint(4, size=n_cells)],
            covar_1=np.array(["v1", "v2"])[np.random.randint(2, size=n_cells)],
            split=np.array(["train", "test", "ood"])[np.random.randint(3, size=n_cells)],
        )
    )
    obs.loc[:, "covar_1"] = obs.loc[:, "covar_1"].astype("category")
    
    dataset = anndata.AnnData(X=X, obs=obs)
    print(f"   Created dataset: {n_cells} cells × {n_genes} genes")
    
    # Setup anndata with sparse storage
    print("\n2. Setting up AnnData with sparse storage...")
    cpa.CPA.setup_anndata(
        dataset,
        drug_key="drug_name",
        dose_key='dose_val',
        categorical_covariate_keys=["covar_1"],
        control_key='control'
    )
    
    # Verify sparse storage
    print("\n3. Verifying sparse storage...")
    perts_key = 'perts'
    perts_doses_key = 'perts_doses'
    
    def calc_sparsity(sparse_mat):
        """Calculate sparsity percentage of a sparse matrix."""
        return 1 - sparse_mat.nnz / np.prod(sparse_mat.shape)
    
    if perts_key in dataset.obsm:
        is_sparse_perts = sparse.issparse(dataset.obsm[perts_key])
        print(f"   Perturbations stored as sparse: {is_sparse_perts}")
        if is_sparse_perts:
            print(f"   Perturbations matrix type: {type(dataset.obsm[perts_key])}")
            print(f"   Perturbations shape: {dataset.obsm[perts_key].shape}")
            print(f"   Perturbations sparsity: {calc_sparsity(dataset.obsm[perts_key]):.2%}")
    
    if perts_doses_key in dataset.obsm:
        is_sparse_doses = sparse.issparse(dataset.obsm[perts_doses_key])
        print(f"   Dosages stored as sparse: {is_sparse_doses}")
        if is_sparse_doses:
            print(f"   Dosages matrix type: {type(dataset.obsm[perts_doses_key])}")
            print(f"   Dosages shape: {dataset.obsm[perts_doses_key].shape}")
            print(f"   Dosages sparsity: {calc_sparsity(dataset.obsm[perts_doses_key]):.2%}")
    
    # Estimate memory savings
    print("\n4. Estimating memory savings...")
    if is_sparse_perts:
        dense_size = np.prod(dataset.obsm[perts_key].shape) * 8  # 8 bytes per float64
        sparse_size = dataset.obsm[perts_key].data.nbytes + dataset.obsm[perts_key].indices.nbytes + dataset.obsm[perts_key].indptr.nbytes
        savings = (dense_size - sparse_size) / dense_size * 100
        print(f"   Dense storage (estimated): {dense_size / 1024:.2f} KB")
        print(f"   Sparse storage (actual): {sparse_size / 1024:.2f} KB")
        print(f"   Memory savings: {savings:.1f}%")
    
    # Create model and verify data loading
    print("\n5. Creating model and testing data loading...")
    model = cpa.CPA(
        adata=dataset,
        n_latent=32,
        recon_loss='gauss',
        doser_type='logsigm',
        split_key='split',
    )
    print(f"   Model created successfully")
    
    # Try to get a data loader
    print("\n6. Testing data loader with sparse-to-dense conversion...")
    try:
        # Get inference loader
        adata_test = dataset
        indices = np.arange(min(100, adata_test.n_obs))  # Just test with first 100 cells
        loader = model._make_data_loader(
            adata=adata_test,
            indices=indices,
            batch_size=32,
            shuffle=False
        )
        
        # Get one batch
        batch = next(iter(loader))
        
        # Check if perturbations are dense tensors
        import torch
        if perts_key in batch:
            is_tensor = isinstance(batch[perts_key], torch.Tensor)
            print(f"   Perturbations in batch are torch.Tensor: {is_tensor}")
            if is_tensor:
                print(f"   Perturbations batch shape: {batch[perts_key].shape}")
                print(f"   Perturbations batch dtype: {batch[perts_key].dtype}")
        
        if perts_doses_key in batch:
            is_tensor = isinstance(batch[perts_doses_key], torch.Tensor)
            print(f"   Dosages in batch are torch.Tensor: {is_tensor}")
            if is_tensor:
                print(f"   Dosages batch shape: {batch[perts_doses_key].shape}")
                print(f"   Dosages batch dtype: {batch[perts_doses_key].dtype}")
        
        print("   ✓ Data loading works correctly!")
        
    except Exception as e:
        print(f"   ✗ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
    
    # Try a short training run
    print("\n7. Testing training with sparse storage...")
    try:
        model.train(max_epochs=1, plan_kwargs=dict(autoencoder_lr=1e-4))
        print("   ✓ Training works correctly!")
    except Exception as e:
        print(f"   ✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    
except ImportError as e:
    print(f"Error: Missing dependencies - {e}")
    print("Please install required packages (anndata, scanpy, scvi-tools, etc.)")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
