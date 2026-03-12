# data/: Datasets

## Structure

```
data/
├── raw/                    Competition-provided data
│   ├── train.csv           14,351 rows × 58 cols (Dec 2022 – Jul 2024)
│   ├── test.csv            5,065 rows × 57 cols (Aug 2024 – Feb 2025)
│   ├── sample_submission.csv   Submission format template
│   ├── plant_metadata.csv  16 Spanish pumped storage units
│   ├── prod_unavailable.csv    Production unavailability per unit per hour
│   └── cons_unavailable.csv    Consumption unavailability per unit per hour
│
└── actuals/
    └── test_actuals.csv    Ground truth from Energy-Charts API (for offline evaluation)
```

## Column Groups (train.csv / test.csv)

| Group | Columns | Description |
|---|---|---|
| **Identifiers** | `id`, `datetime_start` | Unique row ID, UTC hourly timestamp |
| **Target** | `es_total_ps` | Net pumped storage output (MW). Train only. |
| **ES fundamentals** | `es_demand_f_d1/d2`, `es_wind_f_d1/d2`, `es_solar_f_d1/d2`, etc. | Spanish day-ahead and two-day-ahead forecasts |
| **FR/DE/PT fundamentals** | `fr_*`, `de_*`, `pt_*` | Cross-border country forecasts |
| **Hydro** | `es_hydro_ror_f_*`, `es_hydro_res_f_*`, `es_hydro_inflow_f_*`, `es_hydro_balance_f_*` | Run-of-river, reservoir, inflow, balance |
| **Weather** | `es_temp_f_*`, `es_precip_f_*`, `es_wind_speed_f_*` | Temperature, precipitation, wind speed |
| **Prices** | `es_gas_market_price_d1/d2`, `eua_price` | Gas price, EU carbon allowance |
| **Grid** | `fr_es_ntc_d1`, `es_fr_ntc_d1` | Net transfer capacity (FR-ES border) |
| **Normals** | `es_wind_n_d1/d2`, `es_solar_n_d1/d2`, `fr_solar_n_d1/d2` | Seasonal normal forecasts |

## Plant Metadata

16 Spanish pumped storage units with columns: `unit_name`, `unit_eic`, `unit_prod_capacity`, `unit_cons_capacity`, `bidding_zone`.

Fleet totals:
- Production capacity: 5,535.6 MW
- Pumping capacity: 5,252.1 MW

## Notes

- All timestamps are UTC. Spanish local time is CET/CEST (UTC+1/+2).
- `d1` = day-ahead forecast, `d2` = two-day-ahead forecast.
- Unavailability files have one row per unit per hour (sparse, only hours with unavailability > 0).
- No external data is allowed for competition submissions.
