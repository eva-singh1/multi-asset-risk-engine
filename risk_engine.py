import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

class PortfolioRiskEngine:
    def __init__(self, tickers: list, weights: list, initial_portfolio_value: float = 1000000.0):
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0.")
        self.tickers = tickers
        self.weights = np.array(weights)
        self.initial_portfolio_value = initial_portfolio_value
        self.asset_returns = pd.DataFrame()
        self.covariance_matrix = pd.DataFrame()
        self.mean_returns = pd.Series(dtype=float)
        self.metrics_dashboard = {}

    def ingest_market_data(self, start_date: str, end_date: str):
        print(f"Opening live data pipeline for target risk assets: {self.tickers}...")
        price_data = {}
        for ticker in self.tickers:
            asset_obj = yf.Ticker(ticker)
            df = asset_obj.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            price_data[ticker] = df['Close'].astype(float)
        combined_df = pd.DataFrame(price_data).ffill().bfill()
        self.asset_returns = np.log(combined_df / combined_df.shift(1)).dropna()
        self.mean_returns = self.asset_returns.mean()
        self.covariance_matrix = self.asset_returns.cov()

    def calculate_parametric_risk(self, confidence_level: float = 0.99, horizon: int = 1):
        portfolio_variance = np.dot(self.weights.T, np.dot(self.covariance_matrix.values, self.weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        portfolio_expected_return = np.dot(self.weights, self.mean_returns)
        scaled_mu = portfolio_expected_return * horizon
        scaled_sigma = portfolio_volatility * np.sqrt(horizon)
        z_score = norm.ppf(confidence_level)
        var_pct = z_score * scaled_sigma - scaled_mu
        alpha = 1 - confidence_level
        cvar_pct = (alpha**-1 * norm.pdf(z_score) * scaled_sigma) - scaled_mu
        return {"VaR_Nominal": var_pct * self.initial_portfolio_value, "CVaR_Nominal": cvar_pct * self.initial_portfolio_value}

    def calculate_historical_risk(self, confidence_level: float = 0.99, horizon: int = 1):
        portfolio_hist_returns = self.asset_returns.dot(self.weights)
        if horizon > 1:
            portfolio_hist_returns = portfolio_hist_returns.rolling(window=horizon).sum().dropna()
        alpha = 1 - confidence_level
        var_pct = -np.percentile(portfolio_hist_returns, alpha * 100)
        tail_losses = portfolio_hist_returns[portfolio_hist_returns <= -var_pct]
        cvar_pct = -tail_losses.mean() if len(tail_losses) > 0 else var_pct
        return {"VaR_Nominal": var_pct * self.initial_portfolio_value, "CVaR_Nominal": cvar_pct * self.initial_portfolio_value}

    def run_multivariate_monte_carlo(self, simulations: int = 50000, horizon: int = 1, confidence_level: float = 0.99, seed: int = 42):
        np.random.seed(seed)
        num_assets = len(self.tickers)
        sigma_matrix = self.covariance_matrix.values
        mu_vector = self.mean_returns.values
        try:
            L = np.linalg.cholesky(sigma_matrix)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(sigma_matrix + np.eye(num_assets) * 1e-8)
        raw_shocks = np.random.normal(0, 1, size=(num_assets, simulations))
        correlated_shocks = np.dot(L, raw_shocks)
        log_drift = (mu_vector - 0.5 * np.diagonal(sigma_matrix))[:, np.newaxis]
        simulated_log_returns = (log_drift * horizon) + (correlated_shocks * np.sqrt(horizon))
        portfolio_sim_returns = np.dot(self.weights, np.exp(simulated_log_returns) - 1)
        alpha = 1 - confidence_level
        var_pct = -np.percentile(portfolio_sim_returns, alpha * 100)
        sim_tail_losses = portfolio_sim_returns[portfolio_sim_returns <= -var_pct]
        cvar_pct = -sim_tail_losses.mean() if len(sim_tail_losses) > 0 else var_pct
        return {"VaR_Nominal": var_pct * self.initial_portfolio_value, "CVaR_Nominal": cvar_pct * self.initial_portfolio_value}

    def generate_risk_report(self, confidence_level: float = 0.99, horizon: int = 1):
        p_risk = self.calculate_parametric_risk(confidence_level, horizon)
        h_risk = self.calculate_historical_risk(confidence_level, horizon)
        m_risk = self.run_multivariate_monte_carlo(simulations=50000, horizon=horizon, confidence_level=confidence_level)
        self.metrics_dashboard = {
            "Confidence Bounds Interval": f"{confidence_level * 100:.1f}%",
            "Temporal Holding Horizon"  : f"{horizon} Trading Day(s)",
            "Parametric VaR ($)"        : f"${p_risk['VaR_Nominal']:,.2f}",
            "Parametric CVaR ($)"       : f"${p_risk['CVaR_Nominal']:,.2f}",
            "Historical VaR ($)"        : f"${h_risk['VaR_Nominal']:,.2f}",
            "Historical CVaR ($)"       : f"${h_risk['CVaR_Nominal']:,.2f}",
            "Monte Carlo VaR ($)"       : f"${m_risk['VaR_Nominal']:,.2f}",
            "Monte Carlo CVaR ($)"      : f"${m_risk['CVaR_Nominal']:,.2f}"
        }

    def display_risk_assessment(self):
        print("=" * 80)
        print("     INSTITUTIONAL MULTI-ASSET FINANCIAL RISK EXPOSURE ASSESSMENT MODEL ")
        print("=" * 80)
        print(f"Total Portfolio Capital Under Valuation Base: ${self.initial_portfolio_value:,.2f}")
        print("-" * 80)
        print("Cross-Paradigm Tail Value at Risk (VaR) & Expected Shortfall (CVaR) Breakdown:")
        print(f"  [1] Parametric VaR: {self.metrics_dashboard['Parametric VaR ($)']} | CVaR: {self.metrics_dashboard['Parametric CVaR ($)']}")
        print(f"  [2] Historical VaR: {self.metrics_dashboard['Historical VaR ($)']} | CVaR: {self.metrics_dashboard['Historical CVaR ($)']}")
        print(f"  [3] Monte Carlo VaR: {self.metrics_dashboard['Monte Carlo VaR ($)']} | CVaR: {self.metrics_dashboard['Monte Carlo CVaR ($)']}")
        print("=" * 80)

if __name__ == "__main__":
    assets = ["SPY", "TLT", "GLD", "BTC-USD"]
    allocations = [0.45, 0.25, 0.15, 0.15]
    risk_simulator = PortfolioRiskEngine(tickers=assets, weights=allocations, initial_portfolio_value=2000000.0)
    risk_simulator.ingest_market_data(start_date="2021-01-01", end_date="2026-01-01")
    risk_simulator.generate_risk_report(confidence_level=0.99, horizon=1)
    risk_simulator.display_risk_assessment()
