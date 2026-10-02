# Data Access and Release Boundary

This public repository contains the complete final-pipeline source, frozen configuration, non-identifying aggregate cohort statistics, per-seed optimization histories, aggregate evaluation results, manuscript tables, and aggregate figures.

Clinical images, record-level manifests, patient or eye grouping keys, blinded-review source images, feature tensors, model checkpoints, credentials, and execution-session artifacts are not distributed through GitHub. These materials remain in the controlled research environment and require the applicable institutional authorization, ethics approval, data-use agreement, and a secure transfer channel.

The committed `.gitignore` treats the controlled paths and model-weight formats as non-publishable by default. The public aggregate release can be audited against:

- `data/aggregate/development_reviewed_resplit.v1.audit.json`
- `results/multiseed_mean.json`
- `tables/cohort_flow.csv`
- `tables/class_distribution.csv`
- `tables/final_task_metrics.csv`
- `tables/per_seed_metrics.csv`
