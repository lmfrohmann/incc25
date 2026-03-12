# Incommodities Case Crunch: Forecasting Competition

> **Start:** Mar 10, 2025 &nbsp;|&nbsp; **Close:** Mar 12, 2025

---

## Description

### The European Power Market

Europe's power market is a vast, interconnected network spanning multiple countries, each with its own generation mix, regulatory framework, and demand patterns. Thanks to initiatives promoting cross-border collaboration and market integration, electricity can flow more freely between countries than ever before.

At InCommodities, the EU Forward Power team is focused on trading of both futures and capacities. Forward power futures contracts allow us to buy and sell electricity at a predetermined price for delivery at a future date. Capacity trading involves securing the right to transmit power across country borders. By simultaneously participating in these markets, we can optimize our positions, manage risk, and help ensure that adequate generation resources are available when and where they are needed, without ever producing any electricity ourselves.

In order to be successful in this market, we have to be able to accurately model power supply and demand across Europe. For this purpose we need to know exactly how much power of each fuel type will be produced in each hour of the day. Each country has its own mix of power sources, each with varying magnitude of installed capacity and varying marginal costs, this is known as the country's **power mix**.

The power supply stack is visualised as a series of bars ordered from cheapest to most expensive, where:
- **Colour** → fuel type
- **Width** → installed capacity (MWh)
- **Height** → marginal cost

The power stack can vary each hour based on renewable generation, maintenance or outages of plants, and more. Every hour, supply has to meet demand, and **the price of power is set by the most expensive power plant needed to cover demand**.

---

## Hydro Pumped Storage Production

Hydro pumped storage is a special type of hydropower facility that acts like a giant battery for the power grid. These plants typically consist of two reservoirs at different elevations:

- During **low demand / low price** periods → uses electricity to **pump water up** to the upper reservoir.
- During **high demand / high price** periods → releases water through turbines to **generate electricity**.

This behaviour makes it super tricky to forecast, but with an ever-increasing supply of renewable energy it gets ever more crucial to model accurately.

### Why Is Pumped Storage Production So Tricky to Forecast?

#### 1. Price Volatility and Arbitrage Opportunities
Pump storage operators aim to exploit price differentials between off-peak and peak periods. Their decisions hinge on forecasts of future electricity prices, which can swing significantly due to differences in demand and renewables output.

#### 2. Optimal Pumping vs. Producing Decisions
At any given moment, the operator must decide whether to:
- **Pump water up** (incurring a cost) to store potential energy for future high prices
- **Generate electricity** (earning revenue)
- **Stand by** if price signals are not favourable

These decisions require balancing the immediate cost of pumping with the expected revenue from generating power in the future.

#### 3. Water Constraints and Reservoir Management
Reservoir levels are finite, and water usage decisions must account for:
- **Physical Constraints:** Maximum and minimum water levels
- **Regulatory/Environmental Constraints:** Minimum water flows for ecological reasons

#### 4. Forward-Looking Strategies
Pumped storage operators must look several hours, or even days, ahead to optimise their profit. They rely on forecasts for:
- Electricity demand
- Renewable output (wind, solar)
- Fuel prices
- Cross-border flows

#### 5. Market Interactions and Transmission Capacity
European markets are increasingly integrated, meaning prices in one region can be affected by conditions in neighbouring markets. Transmission constraints or interconnector bottlenecks can drastically change the expected profit from pumping or producing.

---

## Your Assignment

As an analyst on the Forward Power team, you've been tasked with **predicting Spanish pumped storage production**. This key resource plays a vital role in stabilising the energy market by balancing renewable energy production and ensuring supply meets demand.

---

## Dataset

The dataset includes five key files:

| File | Description |
|------|-------------|
| `train.csv` | Historical data including the target variable, for training your model |
| `test.csv` | Out-of-sample period data, excluding the target variable |
| `plant_metadata.csv` | Metadata on Spanish pump storage units |
| `cons_unavailable.csv` | Unavailable pumping capacity for a subset of pump storage units |
| `prod_unavailable.csv` | Unavailable production capacity for a subset of pump storage units |

---

## Approach

1. **Access the Data**: Download and familiarise yourself with the dataset structure.
2. **Gather Insights**: Analyse patterns, trends, and correlations influencing pumped storage production.
3. **Propose and Build a Model**: Consider regression models, time-series analysis, ML algorithms, or structural models.
4. **Train Your Model**: Optimise for accuracy and ensure it generalises well.
5. **Generate Predictions**: Apply your model to `test.csv`.
6. **Submit Your Results**: Format submissions per the provided guidelines. Multiple submissions allowed.
7. **Compete and Iterate**: Monitor the leaderboard and refine your strategy.

---

## Evaluation

Submissions are scored using **Root Mean Squared Error (RMSE)**:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

> **Note:** The live leaderboard is based on only **10% of the test data** to prevent overfitting. The final leaderboard is published after the submission deadline.

### Final Scoring Weights

| Criterion | Weight |
|-----------|--------|
| RMSE (final leaderboard) | ~60% |
| Methodology & logic | ~40% |

> A good methodology is **not necessarily the most complex one!**
