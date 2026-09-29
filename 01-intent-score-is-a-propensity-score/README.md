# 01 — Your intent score is a propensity score

Intent scores answer "who is likely to buy." They do not answer "will contacting this account cause a sale." This piece shows the difference on synthetic CRM data where the true effect of outreach on every account is known.

**Read the piece:** [When High Intent Doesn't Matter](writeup/when_high_intent_doesnt_matter.md)

![Sure-things decomposition headline chart](figures/generated/headline_diagnostic_buckets.png)

## Files

| File | What it does |
|------|--------------|
| [`writeup/when_high_intent_doesnt_matter.md`](writeup/when_high_intent_doesnt_matter.md) | The essay. |
| [`src/generate_synthetic_crm.py`](src/generate_synthetic_crm.py) | Builds the synthetic dataset: 20,000 accounts, seed 42, planted segments and true treatment effects. |
| [`notebooks/sure_things_decomposition.ipynb`](notebooks/sure_things_decomposition.ipynb) | Fits the baseline model, builds the diagnostic buckets, validates them against the planted truth, and produces every table and the chart. |
| `figures/generated/headline_diagnostic_buckets.png` | The headline chart, written by the notebook. |

## Reproduce

From this folder:

```bash
pip install -r ../requirements.txt
python src/generate_synthetic_crm.py --summary
jupyter nbconvert --to notebook --execute --inplace notebooks/sure_things_decomposition.ipynb
```

The generator writes `data/generated/synthetic_crm.csv`. The notebook also rebuilds it if it is missing.
