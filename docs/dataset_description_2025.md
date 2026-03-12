# Dataset Description

This dataset consists of hourly data capturing various energy- and weather-related variables for multiple countries and regions.

---

## Files Overview

| File | Description |
|------|-------------|
| `train.csv` | Training data with hourly observations including the target variable |
| `test.csv` | Test data: same structure as `train.csv` but without the target variable |
| `plant_metadata.csv` | Metadata on Spanish pump storage units |
| `prod_unavailable.csv` | Publicly reported unavailability of production turbines |
| `cons_unavailable.csv` | Publicly reported unavailability of pumping capacity |
| `sample_submission.csv` | Required submission format (`id`, `es_total_ps`) |

---

## File Details

### `train.csv`
Hourly observations containing energy- and weather-related measurements from Spain (ES), France (FR), Portugal (PT), and Germany (DE).

**Target variable:** `es_total_ps`: Spanish hydro pumped storage production (MW).
> Negative values indicate net positive consumption (i.e., the plant is pumping water up).

---

### `test.csv`
Same structure as `train.csv` but **without** `es_total_ps`. Your goal is to predict this variable for each row.

---

### `plant_metadata.csv`
Information on individual Spanish pump storage units:

| Column | Description |
|--------|-------------|
| `units` | Number of turbines available for production |
| `pump_units` | Number of pumps available for pumping |
| `pump_efficiency` | Ratio of preserved energy when pumping. E.g. efficiency of 0.75 means 0.75 MWh generated per MWh consumed during pumping |
| `capacity_per_unit` | Maximum production capacity per unit (MW) |
| `pump_load_per_unit` | Maximum MW consumed per unit during pumping |

---

### `prod_unavailable.csv`
Reports of production turbines that are unavailable (e.g. due to breakdowns or maintenance):

| Column | Description |
|--------|-------------|
| `unit` | The affected unit |
| `datetime_start` | Start time of the unavailability |
| `unavailable` | Amount of MW unavailable |

---

### `cons_unavailable.csv`
Reports of pumping capacity that is unavailable:

| Column | Description |
|--------|-------------|
| `unit` | The affected unit |
| `datetime_start` | Start time of the unavailability |
| `unavailable` | Amount of MW unavailable |

---

### `sample_submission.csv`
Required submission format with exactly two columns:
- `id`: integer linking to a row in `test.csv`
- `es_total_ps`: your predicted value

> Warning: Ensure `id` is formatted as **integers**, not strings.

---

## Objective

Predict the **aggregated Spanish pumped storage production** `es_total_ps` for each hour in the test set.

- You may use any subset of the provided features and engineer your own derived features.
- **You are not allowed to use any external data sources.**

---

## Feature Naming Convention

Feature names follow the pattern:

```
[countrycode]_[datatype]_[n/f]
```

- `_n` -> seasonal **normal** (expected value given the time of year)
- `_f` -> **forecast**
- `_d1` -> **day-ahead** (d+1) forecast
- `_d2` -> **two-day-ahead** (d+2) forecast

> All features are generated from the information set available at **10:00 AM CEST** on the day prior to `datetime_start`.

**Example:**
- `es_hydro_res_f_d1` at `2023-01-01 00:00:00+01:00` -> forecast for Jan 1 as seen from Dec 31 at 10:00 AM
- `es_hydro_res_f_d2` at `2023-01-01 00:00:00+01:00` -> forecast for Jan 2 as seen from Dec 31 at 10:00 AM

---

## Feature Descriptions

### Identifiers & Time

| Feature | Description |
|---------|-------------|
| `id` | Unique row identifier |
| `datetime_start` | Timestamp marking the start of the observation (timezone-aware) |

---

### Weather

| Feature | Description |
|---------|-------------|
| `es_temp_f` | Temperature forecast for Spain |
| `es_precip_f` | Precipitation forecast for Spain (mm) |
| `es_wind_speed_f` | Average wind speed forecast for Spain (m/s) |

---

### Hydro

| Feature | Description |
|---------|-------------|
| `pt_hydro_ror_f` | Run-of-river hydro generation forecast: Portugal (MW) |
| `es_hydro_ror_f` | Run-of-river hydro generation forecast: Spain (MW) |
| `fr_hydro_ror_f` | Run-of-river hydro generation forecast: France (MW) |
| `es_hydro_res_f` | Reservoir hydro generation forecast: Spain (MW) |
| `fr_hydro_res_f` | Reservoir hydro generation forecast: France (MW) |
| `es_hydro_balance_f` | Forecast deviation from normal hydro balance (GWh): (Inflows + Storage) - Outflow - Normal |
| `es_hydro_inflow_f` | Forecasted hydro inflow into Spanish reservoirs (GWh) |

---

### Wind

| Feature | Description |
|---------|-------------|
| `es_wind_n` | Seasonal normal wind generation: Spain (MW) |
| `es_wind_f` | Wind generation forecast: Spain (MW) |
| `pt_wind_f` | Wind generation forecast: Portugal (MW) |
| `fr_wind_f` | Wind generation forecast: France (MW) |
| `de_wind_f` | Wind generation forecast: Germany (MW) |

---

### Solar

| Feature | Description |
|---------|-------------|
| `es_solar_n` | Seasonal normal solar generation: Spain (MW) |
| `fr_solar_n` | Seasonal normal solar generation: France (MW) |
| `es_solar_f` | Solar generation forecast: Spain (MW) |
| `pt_solar_f` | Solar generation forecast: Portugal (MW) |
| `fr_solar_f` | Solar generation forecast: France (MW) |
| `de_solar_f` | Solar generation forecast: Germany (MW) |

---

### Demand

| Feature | Description |
|---------|-------------|
| `es_demand_f` | Power consumption forecast: Spain (MW) |
| `pt_demand_f` | Power consumption forecast: Portugal (MW) |
| `fr_demand_f` | Power consumption forecast: France (MW) |
| `de_demand_f` | Power consumption forecast: Germany (MW) |

---

### Cross-Border Capacity

| Feature | Description |
|---------|-------------|
| `fr_es_ntc` | Net Transfer Capacity from France to Spain (MW) |
| `es_fr_ntc` | Net Transfer Capacity from Spain to France (MW) |

> NTCs are published daily by TSOs for the Day Ahead auction.

---

### Market Prices

| Feature | Description |
|---------|-------------|
| `es_gas_market_price` | Iberian gas market price (EUR/MWh) |
| `eua_price` | EU Allowance (EUA) price: cost of emitting 1 tonne CO2e |

---

### Target Variable

| Feature | Description |
|---------|-------------|
| `es_total_ps` | Total Spanish hydro pumped storage production (MW). **Negative = net consumption (pumping mode).** |

---

## Derived Price Notes

### Gas Plant Electricity Cost

A typical gas plant has ~50% efficiency and emits CO2 requiring roughly 1/5 of one EUA per MWh of gas burned. The cost of producing 1 MWh of electricity is:

$$P_{\text{elec}} = \frac{P_{\text{gas}} + 0.2 \cdot P_{\text{EUA}}}{0.5}$$

### Iberian Gas Price Cap (Spain & Portugal)

The EU Commission approved a subsidy for Iberian fossil fuel power plants to lower wholesale electricity prices:

| Period | Price Cap |
|--------|-----------|
| Jun–Dec 2022 | EUR 40/MWh, rising EUR 5/month after initial 6 months |
| Apr 2023 | EUR 56.1/MWh |
| Dec 2023 | EUR 65/MWh (phasing out, aligning with market prices) |

The measure was originally set to expire **31 May 2023**, and was extended to **31 December 2023**.
