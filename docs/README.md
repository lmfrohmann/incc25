# docs/: Competition Documentation

Original competition briefs provided by InCommodities.

## Files

| Document | Description |
|---|---|
| `incommodities_case_crunch_2025.md` | Competition rules, scoring methodology (60% RMSE + 40% write-up), submission format, timeline, and constraints. |
| `dataset_description_2025.md` | Detailed column-by-column description of all datasets, variable definitions, units, and data provenance. |

## Key Rules

- **Metric**: RMSE on hourly `es_total_ps` predictions.
- **Scoring**: 60% RMSE ranking + 40% methodology evaluation.
- **Constraint**: No external data beyond the provided datasets.
- **Submission format**: CSV with columns `id`, `es_total_ps`.
