# CPA Memory Optimization - Quick Reference

> **Full Report:** See [MEMORY_EFFICIENCY_REPORT.md](MEMORY_EFFICIENCY_REPORT.md) for detailed analysis

## 🎯 Quick Stats

| Phase | Memory Reduction | Risk Level | Timeframe |
|-------|------------------|------------|-----------|
| **Phase 1** | 60-70% | Low | 1-2 weeks |
| **Phase 2** | +30-40% | Medium | 3-4 weeks |
| **Phase 3** | +20-30% | Low-Medium | 5-6 weeks |
| **TOTAL** | **80-95%** | Mixed | 6 weeks |

## 🏆 Top 5 Optimizations (by Priority)

### 1. Generator-Based Prediction Accumulation ⭐⭐⭐⭐⭐
- **Savings:** 75% during inference
- **Risk:** None (identical results)
- **Complexity:** Low
- **Where:** `_model.py` - `get_latent_representation()` method

### 2. In-Place Mixup Operations ⭐⭐⭐⭐⭐
- **Savings:** 70% during training
- **Risk:** Very Low (minor numerical differences < 1e-6)
- **Complexity:** Medium
- **Where:** `_module.py` - mixup data augmentation

### 3. Sparse Perturbation Storage ⭐⭐⭐⭐
- **Savings:** 85-90% during setup
- **Risk:** None (identical data, different format)
- **Complexity:** Medium
- **Where:** `_model.py` - `setup_anndata()` method

### 4. Release Unused Latent Representations ⭐⭐⭐⭐
- **Savings:** 70-85% during inference
- **Risk:** None (user-controlled)
- **Complexity:** Low
- **Where:** `_model.py` - `custom_predict()` method

### 5. Reduce Default n_samples ⭐⭐⭐⭐
- **Savings:** 75% during inference
- **Risk:** Moderate (affects variance estimates only)
- **Complexity:** Low
- **Where:** `_model.py` - inference methods default parameters

## 📊 Memory Impact by Phase

### Current Baseline (100K cells, 5K genes)
```
Setup:     ~2.5 GB
Training:  ~4.5 GB GPU
Inference: ~8.0 GB GPU
```

### After Phase 1 (Low-Risk Optimizations)
```
Setup:     ~2.5 GB (no change)
Training:  ~2.5 GB GPU (-44%)
Inference: ~2.0 GB GPU (-75%)
```

### After Phase 2 (Medium-Risk Optimizations)
```
Setup:     ~500 MB (-80%)
Training:  ~1.5 GB GPU (-67%)
Inference: ~1.0 GB GPU (-87%)
```

### After Phase 3 (Advanced Optimizations)
```
Setup:     ~100 MB (-96%)
Training:  ~800 MB GPU (-82%)
Inference: ~500 MB GPU (-94%)
```

## 🔧 Implementation Checklist

### Phase 1: Immediate Wins (Start Here!)
- [ ] Implement generator-based accumulation in `get_latent_representation()`
- [ ] Convert mixup operations to in-place (use `torch.mul(..., out=...)`)
- [ ] Add selective latent computation in `custom_predict()`
- [ ] Enable data prefetching in DataLoader (add `num_workers=4`)
- [ ] Add `torch.cuda.empty_cache()` calls after epochs

### Phase 2: Structural Changes
- [ ] Implement sparse storage for perturbation indices (use `scipy.sparse`)
- [ ] Make category strings lazy (compute on-demand)
- [ ] Add gradient accumulation option to training plan
- [ ] Change default `n_samples` from 20 to 5
- [ ] Add mixed precision training support (FP16/BF16)

### Phase 3: Advanced Features
- [ ] Implement gradient checkpointing for encoder/decoder
- [ ] Add memory-mapped AnnData support (backed mode)
- [ ] Implement streaming predictions to disk
- [ ] Optimize adversarial gradient handling
- [ ] Add auto-batch size tuning

## ⚠️ Model Similarity Expectations

| Optimization Type | Expected Difference | Acceptable? |
|-------------------|---------------------|-------------|
| Storage format changes | 0% (identical) | ✅ Always |
| In-place operations | < 0.0001% | ✅ Always |
| Mixed precision (BF16) | 1-3% | ✅ Usually acceptable |
| Reduced sampling (n=5 vs 20) | 0% (mean), 10-20% (variance) | ⚠️ Use case dependent |
| Gradient checkpointing | 0% | ✅ Always (just slower) |

## 📝 Validation Template

After each optimization:

```python
# 1. Train both models with same seed
model_before.train(seed=42)
model_after.train(seed=42)

# 2. Compare predictions
preds_before = model_before.predict(test_data)
preds_after = model_after.predict(test_data)

# 3. Assert similarity
import numpy as np
np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

# 4. Compare metrics
r2_before = compute_r2(preds_before, ground_truth)
r2_after = compute_r2(preds_after, ground_truth)
assert abs(r2_before - r2_after) < 0.01  # Within 1%
```

## 🚀 Quick Start

To begin optimizing:

1. **Measure baseline:**
   ```python
   from cpa import log_memory_usage
   log_memory_usage("baseline")
   ```

2. **Start with Phase 1 optimization #1:**
   - Edit `_model.py`, function `get_latent_representation()`
   - Replace list accumulation with pre-allocated arrays
   - Test and validate

3. **Measure improvement:**
   ```python
   log_memory_usage("after_optimization_1")
   ```

4. **Repeat for each optimization**

## 📚 Related Files

- **Full Report:** [MEMORY_EFFICIENCY_REPORT.md](MEMORY_EFFICIENCY_REPORT.md)
- **Main Model:** `cpa/_model.py`
- **Training:** `cpa/_task.py`
- **Module:** `cpa/_module.py`
- **API:** `cpa/_api.py`

## 🤝 Contributing

When implementing optimizations:
1. Create a feature branch for each optimization
2. Run validation tests (see template above)
3. Add memory profiling before/after
4. Document any model behavior changes
5. Update this checklist

---

**Questions?** See the full report or open an issue.
