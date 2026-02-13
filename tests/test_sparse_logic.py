"""
Simple unit test to verify sparse matrix logic without full package installation.
"""

import numpy as np
import pandas as pd
from scipy import sparse


def test_sparse_mask_creation_logic():
    """Test the sparse mask creation logic independently."""
    
    # Simulate the data
    n_cells = 1000
    n_genes = 500
    
    # Simulate condition categories
    conditions = np.array(["cond_1", "cond_2", "cond_3"])[np.random.randint(3, size=n_cells)]
    
    # Simulate gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    gene_names_series = pd.Series(gene_names)
    
    # Simulate DEG data
    deg_uns_key_data = {
        "cond_1": [f"gene_{i}" for i in range(0, 50)],      # 50 genes
        "cond_2": [f"gene_{i}" for i in range(100, 200)],   # 100 genes
        "cond_3": [f"gene_{i}" for i in range(300, 350)],   # 50 genes
    }
    
    n_deg_r2 = 10
    
    # Test the new sparse implementation logic
    print("Testing sparse mask creation...")
    
    # 1. Map conditions to integer codes
    codes, uniques = pd.factorize(pd.Series(conditions), sort=True)
    
    print(f"  - Number of unique conditions: {len(uniques)}")
    print(f"  - Conditions: {uniques}")
    
    # 2. Pre-compute sparse masks for unique conditions only
    unique_masks = []
    unique_masks_r2 = []
    
    for cov_cond in uniques:
        if cov_cond in deg_uns_key_data.keys():
            # Create boolean mask (1 for DEGs, 0 for others)
            mask_hvg = gene_names_series.isin(deg_uns_key_data[cov_cond]).astype(int)
            mask_hvg_r2 = gene_names_series.isin(deg_uns_key_data[cov_cond][:n_deg_r2]).astype(int)
        else:
            # Fallback: all genes
            mask_hvg = np.ones(n_genes)
            mask_hvg_r2 = np.ones(n_genes)
            
        unique_masks.append(sparse.csr_matrix(mask_hvg))
        unique_masks_r2.append(sparse.csr_matrix(mask_hvg_r2))
    
    # Stack unique masks
    U_mask = sparse.vstack(unique_masks).tocsr()
    U_mask_r2 = sparse.vstack(unique_masks_r2).tocsr()
    
    print(f"  - Unique mask shape: {U_mask.shape}")
    print(f"  - Unique mask R2 shape: {U_mask_r2.shape}")
    
    # 3. Broadcast to full dataset shape using sparse slicing
    mask_sparse = U_mask[codes, :]
    mask_sparse_r2 = U_mask_r2[codes, :]
    
    print(f"  - Final mask shape: {mask_sparse.shape}")
    print(f"  - Final mask R2 shape: {mask_sparse_r2.shape}")
    
    # Verify correctness
    assert mask_sparse.shape == (n_cells, n_genes), f"Incorrect shape: {mask_sparse.shape}"
    assert mask_sparse_r2.shape == (n_cells, n_genes), f"Incorrect R2 shape: {mask_sparse_r2.shape}"
    assert sparse.issparse(mask_sparse), "Result is not sparse"
    assert sparse.issparse(mask_sparse_r2), "Result R2 is not sparse"
    
    # Verify the values are correct by sampling
    for i, cond in enumerate(uniques):
        # Find cells with this condition
        cells_with_cond = np.where(codes == i)[0]
        if len(cells_with_cond) > 0:
            cell_idx = cells_with_cond[0]
            
            # Get the mask for this cell
            cell_mask = mask_sparse[cell_idx, :].toarray().flatten()
            cell_mask_r2 = mask_sparse_r2[cell_idx, :].toarray().flatten()
            
            # Check that the mask matches the DEGs for this condition
            if cond in deg_uns_key_data:
                expected_degs = deg_uns_key_data[cond]
                for j, gene in enumerate(gene_names):
                    if gene in expected_degs:
                        assert cell_mask[j] == 1, f"Gene {gene} should be marked as DEG for {cond}"
                    else:
                        assert cell_mask[j] == 0, f"Gene {gene} should not be marked as DEG for {cond}"
                
                # Check R2 mask (only first n_deg_r2 genes)
                expected_degs_r2 = expected_degs[:n_deg_r2]
                for j, gene in enumerate(gene_names):
                    if gene in expected_degs_r2:
                        assert cell_mask_r2[j] == 1, f"Gene {gene} should be marked as DEG (R2) for {cond}"
    
    # Calculate sparsity
    sparsity = 1 - mask_sparse.nnz / (mask_sparse.shape[0] * mask_sparse.shape[1])
    sparsity_r2 = 1 - mask_sparse_r2.nnz / (mask_sparse_r2.shape[0] * mask_sparse_r2.shape[1])
    
    print(f"  - Mask sparsity: {sparsity:.2%}")
    print(f"  - Mask R2 sparsity: {sparsity_r2:.2%}")
    
    print("✓ Sparse mask creation logic is correct!")
    
    # Compare memory usage
    dense_mask = mask_sparse.toarray()
    sparse_size = mask_sparse.data.nbytes + mask_sparse.indices.nbytes + mask_sparse.indptr.nbytes
    dense_size = dense_mask.nbytes
    
    print(f"\nMemory comparison:")
    print(f"  - Sparse storage: {sparse_size / 1024:.2f} KB")
    print(f"  - Dense storage: {dense_size / 1024:.2f} KB")
    print(f"  - Memory savings: {(1 - sparse_size / dense_size) * 100:.1f}%")


def test_old_vs_new_implementation():
    """Compare the old and new implementations to ensure they produce the same results."""
    
    print("\nComparing old vs new implementation...")
    
    # Simulate the data
    n_cells = 500
    n_genes = 200
    
    conditions = np.array(["cond_A", "cond_B"])[np.random.randint(2, size=n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    gene_names_series = pd.Series(gene_names)
    
    deg_uns_key_data = {
        "cond_A": [f"gene_{i}" for i in range(0, 30)],
        "cond_B": [f"gene_{i}" for i in range(50, 100)],
    }
    
    # Old implementation (simplified)
    cov_cond_unique = np.unique(conditions)
    cov_cond_map = {}
    
    for cov_cond in cov_cond_unique:
        if cov_cond in deg_uns_key_data.keys():
            mask_hvg = gene_names_series.isin(deg_uns_key_data[cov_cond]).astype(int)
            cov_cond_map[cov_cond] = list(mask_hvg)
        else:
            no_mask = list(np.ones(shape=(n_genes,)))
            cov_cond_map[cov_cond] = no_mask
    
    mask_old = np.vstack(
        np.vectorize(lambda x: cov_cond_map[x], otypes=[np.ndarray])(conditions)
    )
    
    # New implementation
    codes, uniques = pd.factorize(pd.Series(conditions), sort=True)
    unique_masks = []
    
    for cov_cond in uniques:
        if cov_cond in deg_uns_key_data.keys():
            mask_hvg = gene_names_series.isin(deg_uns_key_data[cov_cond]).astype(int)
        else:
            mask_hvg = np.ones(n_genes)
        unique_masks.append(sparse.csr_matrix(mask_hvg))
    
    U_mask = sparse.vstack(unique_masks).tocsr()
    mask_new = U_mask[codes, :].toarray()
    
    # Compare
    assert np.allclose(mask_old, mask_new), "Old and new implementations produce different results!"
    
    print("  ✓ Old and new implementations produce identical results")
    print(f"  - Shape: {mask_old.shape}")
    print(f"  - All values match: {np.array_equal(mask_old, mask_new)}")


if __name__ == "__main__":
    test_sparse_mask_creation_logic()
    test_old_vs_new_implementation()
    print("\n✅ All logic tests passed!")
