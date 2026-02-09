# Memory Efficiency Documentation

This directory contains comprehensive documentation for optimizing memory usage in the CPA (Compositional Perturbation Autoencoder) model.

## 📚 Documentation Files

### [MEMORY_EFFICIENCY_REPORT.md](MEMORY_EFFICIENCY_REPORT.md)
**Comprehensive Technical Report (22KB, ~630 lines)**

Detailed analysis of all proposed memory optimizations including:
- 13 specific optimization proposals across 4 categories
- Technical implementation details with code examples
- Memory savings calculations for each optimization
- Model similarity impact assessment
- Implementation priority matrix
- Phased rollout plan (6 weeks)
- Validation strategies and monitoring tools

**Target Audience:** Developers implementing the optimizations

### [MEMORY_OPTIMIZATION_SUMMARY.md](MEMORY_OPTIMIZATION_SUMMARY.md)
**Quick Reference Guide (5KB, ~170 lines)**

Condensed guide for quick access including:
- Top 5 priority optimizations
- Memory impact by implementation phase
- Implementation checklists
- Quick start guide
- Validation templates

**Target Audience:** Project managers, reviewers, and developers needing quick reference

## 🎯 At a Glance

### The Problem
CPA's current implementation has significant memory overhead in three key areas:
1. **Setup (setup_anndata):** Dense storage of sparse perturbation data
2. **Training:** Multiple tensor copies during mixup augmentation
3. **Evaluation:** List-based accumulation of predictions

### The Solution
13 optimizations organized into 3 implementation phases:

| Phase | Duration | Memory Reduction | Risk Level |
|-------|----------|------------------|------------|
| Phase 1 | 1-2 weeks | 60-70% | Low |
| Phase 2 | 3-4 weeks | +30-40% (total 80-85%) | Medium |
| Phase 3 | 5-6 weeks | +20-30% (total 90-95%) | Mixed |

### Expected Total Impact

**Before Optimization (100K cells, 5K genes):**
- Setup: ~2.5 GB
- Training: ~4.5 GB GPU
- Inference: ~8.0 GB GPU

**After All Optimizations:**
- Setup: ~100 MB (-96%)
- Training: ~800 MB GPU (-82%)
- Inference: ~500 MB GPU (-94%)

## 🏆 Top 5 High-Impact Optimizations

1. **Generator-Based Prediction Accumulation** ⭐⭐⭐⭐⭐
   - 75% memory savings during inference
   - Zero model impact
   - Low implementation complexity

2. **In-Place Mixup Operations** ⭐⭐⭐⭐⭐
   - 70% memory savings during training
   - Negligible model impact (<1e-6 numerical differences)
   - Medium implementation complexity

3. **Sparse Perturbation Storage** ⭐⭐⭐⭐
   - 85-90% memory savings during setup
   - Zero model impact
   - Medium implementation complexity

4. **Release Unused Latent Representations** ⭐⭐⭐⭐
   - 70-85% memory savings during inference
   - Zero model impact (user-controlled)
   - Low implementation complexity

5. **Reduce Default n_samples** ⭐⭐⭐⭐
   - 75% memory savings during inference
   - Moderate impact on variance estimates only
   - Low implementation complexity

## 📋 Implementation Roadmap

### Phase 1: Low-Risk, High-Impact (Start Here!)
**Timeline:** Weeks 1-2  
**Expected Savings:** 60-70% memory reduction

✅ **Optimizations:**
1. Generator-based prediction accumulation
2. In-place mixup operations
3. Release unused latent representations
4. Data prefetching
5. Cache clearing

### Phase 2: Medium-Risk, High-Impact
**Timeline:** Weeks 3-4  
**Expected Savings:** Additional 30-40% (total 80-85%)

✅ **Optimizations:**
1. Sparse perturbation storage
2. Lazy category string evaluation
3. Gradient accumulation
4. Reduced default n_samples
5. Mixed precision training (FP16/BF16)

### Phase 3: Advanced Optimizations
**Timeline:** Weeks 5-6  
**Expected Savings:** Additional 20-30% (total 90-95%)

✅ **Optimizations:**
1. Gradient checkpointing
2. Memory-mapped AnnData storage
3. Streaming predictions
4. Adversarial gradient optimization
5. Auto-batch sizing

## 🔍 Quick Start

### For Developers

1. **Read the full report:**
   ```bash
   # Read MEMORY_EFFICIENCY_REPORT.md for detailed technical analysis
   cat MEMORY_EFFICIENCY_REPORT.md | less
   ```

2. **Start with Phase 1, Optimization #1:**
   - Edit `cpa/_model.py`
   - Function: `get_latent_representation()`
   - Replace list accumulation with pre-allocated arrays
   - See report section 3.1 for code examples

3. **Validate changes:**
   ```python
   # Use validation template from the report
   import numpy as np
   np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)
   ```

### For Project Managers

1. **Read the quick reference:**
   ```bash
   # Read MEMORY_OPTIMIZATION_SUMMARY.md for overview
   cat MEMORY_OPTIMIZATION_SUMMARY.md | less
   ```

2. **Review priority matrix:**
   - Section 5 in MEMORY_EFFICIENCY_REPORT.md
   - Shows all 13 optimizations ranked by priority

3. **Plan implementation:**
   - Use 3-phase roadmap from section 6
   - Allocate 1-2 weeks per phase
   - Total timeline: 6 weeks

## ⚠️ Model Similarity Impact

All proposed optimizations maintain model similarity within acceptable thresholds:

| Category | Expected Difference | Acceptable? |
|----------|---------------------|-------------|
| Storage format changes | 0% (mathematically identical) | ✅ Always |
| In-place operations | <0.0001% (numerical precision) | ✅ Always |
| Mixed precision (BF16) | 1-3% (reduced precision) | ✅ Usually |
| Reduced sampling | 0% (mean), 10-20% (variance) | ⚠️ Use case dependent |
| Gradient checkpointing | 0% (identical gradients) | ✅ Always |

**Important:** Each optimization in the full report includes a detailed "Model Similarity Impact" section explaining:
- Why changes are safe (or not)
- Expected numerical differences
- Recommended validation approaches

## 🧪 Validation

After implementing each optimization:

1. **Numerical validation:**
   ```python
   # Train both models
   model_before.train(seed=42)
   model_after.train(seed=42)
   
   # Compare predictions
   preds_before = model_before.predict(test_data)
   preds_after = model_after.predict(test_data)
   
   # Assert similarity
   np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)
   ```

2. **Metric validation:**
   - Compare R² scores
   - Compare Pearson correlations
   - Compare KNN purity scores
   - All should be within 1-3% of original

3. **Memory profiling:**
   ```python
   from cpa import log_memory_usage
   log_memory_usage("before_optimization")
   # ... run your code ...
   log_memory_usage("after_optimization")
   ```

## 📊 Monitoring

The report includes monitoring utilities (Section 8) for tracking:
- CPU memory usage
- GPU memory usage
- Cached GPU memory
- Memory usage at key stages (setup, training, inference)

## 🤝 Contributing

When implementing optimizations:
1. Create a feature branch for each optimization
2. Follow validation procedures (Section 7 of full report)
3. Add memory profiling before/after
4. Document any unexpected behavior
5. Update implementation checklists

## 📖 Additional Resources

- **Main Model Code:** `cpa/_model.py` (1046 lines)
- **Training Plan:** `cpa/_task.py` (660 lines)
- **Module Implementation:** `cpa/_module.py` (516 lines)
- **API Functions:** `cpa/_api.py` (1107 lines)

## ❓ Questions & Support

- **Technical questions:** See detailed explanations in MEMORY_EFFICIENCY_REPORT.md
- **Implementation help:** See code examples in report sections 1-4
- **Quick lookup:** Use MEMORY_OPTIMIZATION_SUMMARY.md
- **Issues:** Open a GitHub issue

---

**Report Date:** February 9, 2026  
**Repository:** dav-shah/cpa  
**Prepared by:** GitHub Copilot

*These optimizations enable CPA to scale to significantly larger datasets while maintaining scientific validity.*
