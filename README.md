# Multi-Asset Portfolio Risk Simulation & Modeling Engine

An independent quantitative portfolio risk platform built to analyze multi-asset volatility, cross-correlation dependencies, and distribution variances across divergent asset classes using parametric, empirical, and stochastic simulations.

## Project Technical Matrix Capabilities
* **Correlated Multivariate Monte Carlo Simulation:** Implements a Cholesky Decomposition algorithm ($L \cdot L^T = \Sigma$) to transform orthogonal random normal Gaussian paths into integrated asset vectors matching empirical market correlation structures.
* **Risk Triangulation Architectures:** Runs three concurrent risk modeling methodologies—Parametric, Historical Simulation, and Monte Carlo frameworks.
* **Expected Shortfall Evaluation (CVaR):** Models tail risk thresholds by computing conditional averages past the Value at Risk percentile floor to capture deep tail loss exposures.
