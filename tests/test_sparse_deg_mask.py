import anndata
import numpy as np
import pandas as pd
from scipy import sparse

import cpa


def generate_synth_with_deg():
    """Generate synthetic data with DEG information for testing sparse mask optimization."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    n_cells = 2000
    n_genes = 500
    X = np.random.randint(low=0, high=1000, size=(n_cells, n_genes))
    
    # Create gene names
    var_names = [f"gene_{i}" for i in range(n_genes)]
    
    obs = pd.DataFrame(
        dict(
            drug_name=np.array(["d1", "d2", "d3", "control"])[np.random.randint(4, size=n_cells)],
            dose_val=np.array([0.1, 0.05, 0.5, 1.0])[np.random.randint(4, size=n_cells)],
            covar_1=np.array(["v1", "v2"])[np.random.randint(2, size=n_cells)],
            covar_2=np.random.randint(3, size=n_cells),
            split=np.array(["train", "test", "ood"])[np.random.randint(3, size=n_cells)],
        )
    )
    obs.loc[:, "covar_1"] = obs.loc[:, "covar_1"].astype("category")
    obs.loc[:, "covar_2"] = obs.loc[:, "covar_2"].astype("category")

    dataset = anndata.AnnData(
        X=X,
        obs=obs,
    )
    dataset.var_names = var_names
    
    # Create a category column for DEG mapping
    dataset.obs["deg_category"] = dataset.obs["covar_1"].astype(str) + "_" + dataset.obs["covar_2"].astype(str)
    
    # Create mock DEG data in uns
    deg_dict = {}
    unique_categories = dataset.obs["deg_category"].unique()
    for cat in unique_categories:
        # Each category has a different subset of genes as DEGs
        n_degs = np.random.randint(50, 150)
        deg_dict[cat] = np.random.choice(var_names, size=n_degs, replace=False).tolist()
    
    dataset.uns["rank_genes_groups_cov"] = deg_dict
    
    return dataset


def test_sparse_deg_mask_storage():
    """Test that DEG masks are stored as sparse matrices."""
    dataset = generate_synth_with_deg()
    
    cpa.CPA.setup_anndata(
        dataset,
        perturbation_key="drug_name",
        control_group="control",
        dosage_key='dose_val',
        categorical_covariate_keys=["covar_1", "covar_2"],
        deg_uns_key="rank_genes_groups_cov",
        deg_uns_cat_key="deg_category",
    )
    
    # Check that DEG masks are stored as sparse matrices
    assert "deg_mask" in dataset.obsm, "DEG mask not found in obsm"
    assert "deg_mask_r2" in dataset.obsm, "DEG mask R2 not found in obsm"
    
    deg_mask = dataset.obsm["deg_mask"]
    deg_mask_r2 = dataset.obsm["deg_mask_r2"]
    
    # Verify they are sparse matrices
    assert sparse.issparse(deg_mask), "DEG mask is not sparse"
    assert sparse.issparse(deg_mask_r2), "DEG mask R2 is not sparse"
    
    # Verify the shape is correct
    assert deg_mask.shape == (dataset.n_obs, dataset.n_vars), f"DEG mask has incorrect shape: {deg_mask.shape}"
    assert deg_mask_r2.shape == (dataset.n_obs, dataset.n_vars), f"DEG mask R2 has incorrect shape: {deg_mask_r2.shape}"
    
    # Verify the data is binary (0 or 1)
    deg_mask_dense = deg_mask.toarray()
    assert np.all(np.isin(deg_mask_dense, [0, 1])), "DEG mask contains non-binary values"
    
    print("✓ DEG masks are correctly stored as sparse matrices")
    print(f"  - DEG mask shape: {deg_mask.shape}")
    print(f"  - DEG mask sparsity: {1 - deg_mask.nnz / (deg_mask.shape[0] * deg_mask.shape[1]):.2%}")
    print(f"  - DEG mask R2 sparsity: {1 - deg_mask_r2.nnz / (deg_mask_r2.shape[0] * deg_mask_r2.shape[1]):.2%}")


def test_model_with_sparse_deg_mask():
    """Test that the model can be trained with sparse DEG masks."""
    dataset = generate_synth_with_deg()
    
    cpa.CPA.setup_anndata(
        dataset,
        perturbation_key="drug_name",
        control_group="control",
        dosage_key='dose_val',
        categorical_covariate_keys=["covar_1", "covar_2"],
        deg_uns_key="rank_genes_groups_cov",
        deg_uns_cat_key="deg_category",
    )
    
    # Create and train model
    model = cpa.CPA(
        adata=dataset,
        n_latent=64,
        recon_loss='gauss',
        doser_type='logsigm',
        split_key='split',
    )
    
    # Train for a few epochs to ensure no errors
    model.train(max_epochs=2, plan_kwargs=dict(autoencoder_lr=1e-4))
    
    # Try prediction
    model.predict(batch_size=512)
    
    print("✓ Model successfully trained and predicted with sparse DEG masks")


if __name__ == "__main__":
    print("Testing sparse DEG mask storage...")
    test_sparse_deg_mask_storage()
    print("\nTesting model training with sparse DEG masks...")
    test_model_with_sparse_deg_mask()
    print("\n✅ All tests passed!")
