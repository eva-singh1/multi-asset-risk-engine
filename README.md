# Multi-Asset Portfolio Risk Simulation & Modeling Engine

An institutional-grade portfolio risk management and quant analytics platform implemented in Python. This standalone engine ingests live, synchronous historical market data streams across divergent asset classes to evaluate portfolio volatility, map asset correlation dependencies, and model extreme tail-risk exposure under multiple quantitative paradigms.

## 📊 Project Overview & Architecture

Hedge funds and institutional risk desks do not rely on a single mathematical framework to evaluate market risk. This engine addresses that paradigm by executing **three concurrent risk modeling suites** to triangulate an asset cross-sectional risk profile, computing both **Value at Risk (VaR)** and **Expected Shortfall (Conditional VaR / CVaR)** at strict statistical confidence thresholds.

The application incorporates a robust data engineering framework that dynamically handles timeline synchronization across international exchanges, shifting raw security close vectors into continuous stationary logarithmic returns to maintain strict distribution integrity.

## 📐 Mathematical Methodologies

### 1. Parametric (Variance-Covariance) Model
* Assumes joint portfolio returns conform to a multivariate normal distribution.
* Utilizes matrix calculus ($\omega^T \Sigma \omega$) to evaluate integrated portfolio variance and volatility parameters.
* Computes analytical VaR boundaries using the inverse cumulative normal distribution ($Z$-score scaling) and projects values across temporal horizons using the Brownian square-root-of-time scaling rule.

### 2. Non-Parametric Historical Simulation
* Bypasses parametric distribution bounds by evaluating empirical historical market paths directly.
* Preserves authentic market fat-tails, negative skewness, high kurtosis, and structural correlation regimes that linear models typically smooth over.
* Defines VaR directly as the explicit empirical percentile loss boundary.

### 3. Correlated Multivariate Monte Carlo Simulation (50,000 Paths)
* Generates 50,000 distinct daily stochastic pricing pathways.
* **Cholesky Decomposition Algorithm:** Implements matrix factorization ($L \cdot L^T = \Sigma$) on the covariance matrix to isolate the lower triangular matrix ($L$). Multiplying independent standard normal Gaussian shocks by this triangle forces independent variables to mirror real-world cross-asset correlation links.
* Projects paths forward via stochastic geometric drift layers before aggregating vectors back into portfolio performance spaces.

### 4. Expected Shortfall / Tail Risk Refinement (CVaR)
* Calculates Conditional Value at Risk to analyze extreme tail risk.
* Measures the absolute mathematical expected mean value of a tail loss *past* the standard VaR boundary floor, answering the critical question: *"If a catastrophic market deviation breaks our risk floor, what is the expected scale of that loss?"*

---

## 🛠️ System Prerequisites & Installation

Ensure you have a modern Python 3 environment active on your system. Install the required quantitative data science and optimization dependencies via your terminal:

```bash
pip3 install numpy pandas yfinance scipy
```

---

## 💻 Step-by-Step Terminal Execution Guide

To run this risk model locally on your machine, follow these command steps:

1. **Open Your Terminal** and navigate into your dedicated repository directory:
   ```bash
   cd ~/multi-asset-risk-engine
   ```

2. **Verify File Existence** to ensure your main script is present:
   ```bash
   ls
   # You should see: risk_engine.py README.md
   ```

3. **Execute the Script**:
   Run the engine pipeline using Python 3:
   ```bash
   python3 risk_engine.py
   ```

---

## 📈 What to Expect as a Result

Upon running the file, the data pipeline activates immediately. The engine reaches out to the live Yahoo Finance API, pulls 5 years of daily closing candles for Equities (**SPY**), Fixed Income Bonds (**TLT**), Commodities (**GLD**), and Digital Assets (**BTC-USD**), auto-aligns their calendar dates, and processes the risk models.

You will see the data ingestion status logs followed by a beautifully formatted text dashboard printed right into your terminal:

```text
Opening live data pipeline for target risk assets: ['SPY', 'TLT', 'GLD', 'BTC-USD']...
Data layer synchronized. Processed 1258 historical market intervals.
================================================================================
     INSTITUTIONAL MULTI-ASSET FINANCIAL RISK EXPOSURE ASSESSMENT MODEL 
================================================================================
Asset Class Allocation Distribution Weights Matrix:
  • Asset Ticker Class Symbol: SPY        | Target Portfolio Allocation: 45.00%
  • Asset Ticker Class Symbol: TLT        | Target Portfolio Allocation: 25.00%
  • Asset Ticker Class Symbol: GLD        | Target Portfolio Allocation: 15.00%
  • Asset Ticker Class Symbol: BTC-USD    | Target Portfolio Allocation: 15.00%
Total Portfolio Capital Under Valuation Base: $2,000,000.00
--------------------------------------------------------------------------------
Risk Window Evaluation Baseline Configurations:
  Target Confidence Threshold Boundary: 99.0%
  Target Temporal Volatility Horizon  : 1 Trading Day(s)
--------------------------------------------------------------------------------
Cross-Paradigm Tail Value at Risk (VaR) & Expected Shortfall (CVaR) Breakdown:
  [1] Parametric (Variance-Covariance) Framework Matrix:
      Value at Risk (VaR Floor)       : $42,156.42
      Expected Shortfall Tail Vector  : $48,312.18
  [2] Empirical Historical Simulation Framework Matrix:
      Value at Risk (VaR Floor)       : $46,892.11
      Expected Shortfall Tail Vector  : $68,415.53
  [3] Correlated Multivariate Monte Carlo Simulation (N=50,000 Paths):
      Value at Risk (VaR Floor)       : $42,341.05
      Expected Shortfall Tail Vector  : $51,894.27
================================================================================
```
