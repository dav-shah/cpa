# Implementation Plan: Parallel Encoder for CPA Latent Consistency

**Goal:** Regularize the CPA latent space by enforcing a consistency constraint where a new "Parallel Encoder" ($E_{para}$) explicitly maps perturbed cells to their corresponding composed latent state ($z_{after}$).

---

## 1. Where to define the ParallelEncoder class

The CPA architecture relies heavily on `scvi-tools` under the hood. The core network components are primarily defined in `cpa/_utils.py` and `cpa/_module.py`.

* **Recommendation**: You **do not** need to define an entirely new PyTorch class. You can reuse the existing `VanillaEncoder` (or `scvi.nn.Encoder`) which acts as the basal encoder.
* If you prefer custom layers, define `class ParallelEncoder(nn.Module):` in `cpa/_utils.py`.
* In `cpa/_module.py`, you will instantiate this during the `CPAModule` initialization.

## 2. Modifying the CPA Model Class (`cpa/_module.py`)

The main PyTorch module is `CPAModule`. You will need to introduce the parallel encoder and its forward pass computations.

**Changes required in `CPAModule.__init__`:**
```python
# Introduce a toggle flag
self.use_parallel_encoder = use_parallel_encoder
self.parallel_penalty_weight = parallel_penalty_weight

if self.use_parallel_encoder:
    # Initialize separate instance with the SAME dimensions as the main encoder
    self.parallel_encoder = VanillaEncoder(
        n_input=n_genes,
        n_output=n_latent,
        n_hidden=n_hidden_encoder,
        n_layers=n_layers_encoder,
        use_batch_norm=use_batch_norm_encoder,
        # ... match existing encoder kwargs
    )
```

**Changes required in `CPAModule.inference` (or forward pass):**
During the standard inference step, the model computes $z_{basal}$ and later structures $z_{after}$ by adding $\delta_p$ and $\delta_c$.
Capture the input array `x` explicitly (the perturbed gene expressions) and pass it through the new encoder:
```python
# standard forward step
z_basal = self.encoder(x)
delta_p = ... # computed perturbation embeddings
delta_c = ... # computed covariate embeddings
z_after = z_basal + delta_p + delta_c

# ADD PARALLEL PATH
z_parallel = None
if self.use_parallel_encoder and self.training:
    # We only apply it during training as a regularizer
    z_parallel = self.parallel_encoder(x)

# Expose both representations to the loss function
inference_outputs["z_after"] = z_after
inference_outputs["z_parallel"] = z_parallel
```

## 3. Updating the Loss Function in the Training Loop

The loss computation needs to absorb an MSE penalty pushing $z_{parallel} \to z_{after}$. This involves two files.

**Step A: Modifying `CPAModule.loss` in `cpa/_module.py`**
Compute the MSE distance between the composed vectors and the explicitly predicted ones.

```python
def loss(self, tensors, inference_outputs, generative_outputs):
    recon_loss = ...
    kl_loss = ...
    
    # ADDED PARALLEL LOSS
    parallel_loss = torch.tensor(0.0, device=recon_loss.device)
    if self.use_parallel_encoder and "z_parallel" in inference_outputs and inference_outputs["z_parallel"] is not None:
        z_after = inference_outputs["z_after"]
        z_parallel = inference_outputs["z_parallel"]
        
        # Detach z_after if you ONLY want to train the parallel encoder, 
        # or leave attached if you want the main network to be regularized by E_para.
        parallel_loss = torch.nn.functional.mse_loss(z_parallel, z_after)

    return recon_loss, kl_loss, parallel_loss
```

**Step B: Modifying `training_step` in `cpa/_task.py`**
In your Lightning task module (which inherits standard steps), ingest the third component of the loss:

```python
def training_step(self, batch, batch_idx):
    # ...
    recon_loss, kl_loss, parallel_loss = self.module.loss(...)
    
    loss = recon_loss + kl_loss + (self.module.parallel_penalty_weight * parallel_loss)
    # ... Add parallel_loss to self.log() metrics
```
*(Optionally: if you need to update the `CPA` core API class in `cpa/_model.py` to accept the `use_parallel_encoder` args and pass them down to `CPAModule`.)*

## 4. Architectural Suggestion: Shared vs. Separate Weights

**TL;DR:** Use **separate** weights (`self.parallel_encoder` distinct from `self.encoder`).

**Detailed explanation:**
The fundamental goal of the existing base encoder ($\text{Encoder}_{basal}$) in CPA is to learn a representation that is **invariant** to perturbations and covariates. It accomplishes this mapping $X_{perturbed} \to z_{basal}$ via adversarial penalties that try to scrub the perturbation signals out of the latent space.

Conversely, the goal of your proposed $E_{para}$ is precisely the opposite: it must map $X_{perturbed} \to z_{after}$, which **explicitly retains** the perturbation and covariate signals.
If you were to share weights between these encodings, the network would suffer from conflicting gradients: the adversarial critic would be forcing the shared layers to forget the perturbation, while your new MSE parallel loss would be implicitly forcing the shared layers to retain enough information to reconstruct the perturbation in $z_{after}$.

Using a separate network (with the same architecture dimensions) sidesteps this constraint perfectly, allowing $E_{para}$ to maintain the regularizing geometry you desire without destroying CPA's disentanglement capabilities.
