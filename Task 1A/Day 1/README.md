# Requirement tiers

Installed in this order, matching the day-by-day breakdown, instead of one
large upfront install:

1. `core.txt` - Day 1-4 (data, classical ML, frequentist HMM, changepoint)
2. `bayesian.txt` - Day 5-6 (PyMC / NumPyro / ArviZ)
3. `deep_learning.txt` - Day 7-8 (TensorFlow, TensorFlow Probability, Torch,
   PyG, SHAP)
4. `foundation_models.txt` - Day 8 (Chronos, TimesFM; see note in file)
5. `conformal_tda.txt` - Day 3 (TDA features) and Day 11 (conformal
   prediction)

Install a tier with:

```
pip install -r requirements/<tier>.txt
```

`environment.yml` at the repo root lists the same set for conda users.

**Status as of Day 3:** `core.txt` fully installed. `giotto-tda` (from
`conformal_tda.txt`) and `torch`/`torch_geometric` (from
`deep_learning.txt`) were pulled forward and installed early, since Day
3's TDA and sector-GCN features need them now rather than on Day 8/11.
See the notes inside those two files for how torch was installed
CPU-only to avoid pulling a multi-GB CUDA stack, and the numpy/
scikit-learn version downgrade giotto-tda caused. `bayesian.txt`,
`tensorflow` (from `deep_learning.txt`), and `foundation_models.txt` are
still not installed.
