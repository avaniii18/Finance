import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_plot(data_series, axes, title_prefix, ylabel):
    # Formulate a simple discrete dynamical system (AR(1) process):
    # Dynamical Rule: x(t) = a * x(t-1) + b
    x_t_minus_1 = data_series.values[:-1]
    x_t = data_series.values[1:]

    # Use numpy polyfit to find the parameters 'a' (slope) and 'b' (intercept)
    a, b = np.polyfit(x_t_minus_1, x_t, 1)
    
    print(f"\n[{title_prefix}] Estimated Dynamical System:")
    print(f"x(t) = {a:.4f} * x(t-1) + {b:.4f}")
    
    # Calculate the fixed point (equilibrium) where x(t) = x(t-1) = x*
    fixed_point = None
    if abs(a - 1) > 1e-5:
        fixed_point = b / (1 - a)
        print(f"Fixed Point (Equilibrium): {fixed_point:.6f}")
    else:
        print("System is a random walk (a ~ 1), no single fixed point.")
    
    # Generate the deterministic part of the system
    x_t_predicted = a * x_t_minus_1 + b

    # Subplot 1: Time series
    dates = data_series.index[1:]
    axes[0].plot(dates, x_t, label=f'Actual {ylabel} $x(t)$', alpha=0.6, color='blue')
    axes[0].plot(dates, x_t_predicted, label='System Model Output', alpha=0.8, color='red', linestyle='--')
    axes[0].set_title(f'Time Series: {title_prefix}')
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Phase space
    axes[1].scatter(x_t_minus_1, x_t, alpha=0.5, color='purple', s=15, label='State Transitions')
    
    # Plot the line of best fit (our dynamical system mapping)
    x_range = np.linspace(min(x_t_minus_1), max(x_t_minus_1), 100)
    axes[1].plot(x_range, a * x_range + b, color='red', label=f'Mapping: x(t) = {a:.3f}x(t-1) + {b:.3f}')
    
    # Plot y = x line
    axes[1].plot(x_range, x_range, color='black', linestyle=':', label='Identity (y=x)')
    
    if fixed_point is not None and min(x_t_minus_1) <= fixed_point <= max(x_t_minus_1):
        axes[1].scatter([fixed_point], [fixed_point], color='green', s=100, zorder=5, label=f'Fixed Point ({fixed_point:.4f})')
        
    axes[1].set_title(f'Phase Portrait: {title_prefix}')
    axes[1].set_xlabel('State at t-1: $x(t-1)$')
    axes[1].set_ylabel('State at t: $x(t)$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

def main():
    ticker = "AAPL"
    print(f"Downloading 1 year of {ticker} stock data...")
    data = yf.download(ticker, period="1y", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'][ticker]
    else:
        close_prices = data['Close']

    close_prices = close_prices.dropna()
    returns = np.log(close_prices / close_prices.shift(1)).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Analysis 1: Log Returns
    analyze_and_plot(returns, axes[0], f"{ticker} Log Returns", "Log Return")
    
    # Analysis 2: Actual Prices
    analyze_and_plot(close_prices, axes[1], f"{ticker} Actual Prices", "Price (USD)")

    plt.tight_layout()
    output_img = 'dynamical_system_phase_portrait.png'
    plt.savefig(output_img, dpi=300)
    print(f"\nPlot saved successfully as '{os.path.abspath(output_img)}'")

if __name__ == "__main__":
    main()
