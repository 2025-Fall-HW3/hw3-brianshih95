"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01",
                      end="2024-04-01", auto_adjust=False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=150, gamma=-5, momentum_period=120):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        self.momentum_period = momentum_period

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        # Strategy: Momentum + Mean-Variance Optimization with Risk Management
        for i in range(max(self.lookback, self.momentum_period) + 1, len(self.price)):
            # Calculate momentum scores (cumulative return over momentum_period)
            momentum_window = self.returns[assets].iloc[i - self.momentum_period: i]
            momentum_scores = (1 + momentum_window).prod() - 1

            # Select top performing assets (top 60%)
            n_select = max(int(len(assets) * 0.6), 3)
            top_assets = momentum_scores.nlargest(n_select).index.tolist()

            # Get recent returns for selected assets
            R_n = self.returns[top_assets].iloc[i - self.lookback: i]

            # Calculate covariance and mean returns
            Sigma = R_n.cov().values
            mu = R_n.mean().values
            n = len(top_assets)

            # Optimize portfolio using Mean-Variance with Gurobi
            weights = self.mv_opt_constrained(
                Sigma, mu, n, top_assets, momentum_scores)

            # Assign weights
            for j, asset in enumerate(top_assets):
                self.portfolio_weights.loc[self.price.index[i],
                                            asset] = weights[j]

            # Set non-selected assets to 0
            for asset in assets:
                if asset not in top_assets:
                    self.portfolio_weights.loc[self.price.index[i], asset] = 0

        # Set excluded asset to 0
        self.portfolio_weights[self.exclude] = 0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt_constrained(self, Sigma, mu, n, top_assets, momentum_scores):
        """
        Mean-Variance Optimization with momentum tilt
        """
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                # Initialize Decision Variables w (portfolio weights)
                w = model.addMVar(n, name="w", lb=0, ub=0.35)   # Max 35% per asset

                # Momentum adjustment: favor assets with higher momentum
                momentum_adj = np.array([momentum_scores[asset]
                                        for asset in top_assets])
                momentum_adj = momentum_adj / \
                    np.abs(momentum_adj).max()  # Normalize
                adjusted_mu = mu + 0.1 * momentum_adj  # Add momentum bonus

                # Define the objective function: maximize w^T * mu - (gamma/2) * w^T * Sigma * w
                portfolio_return = w @ adjusted_mu
                portfolio_variance = w @ Sigma @ w

                # Set objective: maximize expected return - (gamma/2) * variance
                model.setObjective(
                    portfolio_return - (self.gamma / 2) * portfolio_variance,
                    gp.GRB.MAXIMIZE
                )

                # Add constraint: sum of weights = 1 (fully invested)
                model.addConstr(w.sum() == 1, name="budget")

                # Optimize
                model.optimize()

                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    solution = [model.getVarByName(
                        f"w[{i}]").X for i in range(n)]
                    return solution
                else:
                    raise Exception("Optimization failed")

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
