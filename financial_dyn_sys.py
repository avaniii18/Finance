import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def koopman_dmd_analysis(data_series, axes, title_prefix, ylabel, delay_dim=10, rank=5):
    """
    Applies Hankel Dynamic Mode Decomposition (Hankel-DMD) to approximate 
    the Koopman Operator for a 1D financial time series.
    """
    data = data_series.values
    N = len(data)
    
    if N < delay_dim + 2:
        print("Not enough data for this delay dimension.")
        return

    # 1. Time-delay embedding to construct Hankel matrix (observables)
    X = np.array([data[i:N-delay_dim+i+1] for i in range(delay_dim)])
    
    # Snapshot matrices
    X_1 = X[:, :-1]
    X_2 = X[:, 1:]
    
    # 2. Singular Value Decomposition (SVD) on X_1
    U, S, Vh = np.linalg.svd(X_1, full_matrices=False)
    
    # Truncate to desired rank
    r = min(rank, len(S))
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh[:r, :].T
    
    # 3. Approximate the Koopman Operator (K_tilde) in reduced space
    K_tilde = Ur.T @ X_2 @ Vr @ np.linalg.inv(Sr)
    
    # 4. Koopman Eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(K_tilde)
    
    # 5. Reconstruct state transitions using the full Koopman approximation
    X_2_pred = Ur @ K_tilde @ Ur.T @ X_1
    
    # We plot the latest point in the delay vector (current time step)
    actual_current_state = X_2[-1, :]
    pred_current_state = X_2_pred[-1, :].real
    
    # Time vector for plotting
    dates = data_series.index[delay_dim:]
    
    # Subplot 1: Time series
    axes[0].plot(dates, actual_current_state, label=f'Actual {ylabel}', alpha=0.6, color='blue')
    axes[0].plot(dates, pred_current_state, label='Koopman DMD Output', alpha=0.8, color='red', linestyle='--')
    axes[0].set_title(f'Time Series: {title_prefix}')
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Koopman Spectrum (Eigenvalues on Complex Plane)
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(np.cos(theta), np.sin(theta), color='black', linestyle=':', label='Unit Circle')
    axes[1].scatter(eigenvalues.real, eigenvalues.imag, color='purple', s=50, label='Koopman Eigenvalues', zorder=5)
    
    # Add stability region lines
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Koopman Spectrum: {title_prefix}')
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    print(f"\n[{title_prefix}] Koopman Operator Analysis (Hankel-DMD):")
    print(f"  Time-delay dimension (Observables): {delay_dim}")
    print(f"  Truncated rank: {r}")
    
    # Sort eigenvalues by magnitude
    sorted_idx = np.argsort(-np.abs(eigenvalues))
    top_evals = eigenvalues[sorted_idx][:3]
    print(f"  Top Koopman Eigenvalues:")
    for i, ev in enumerate(top_evals):
        mag = np.abs(ev)
        stability = "Growing" if mag > 1.001 else ("Decaying" if mag < 0.999 else "Stable/Oscillatory")
        print(f"    {i+1}: {ev:.4f}  (Magnitude: {mag:.4f} -> {stability})")

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

    # Analysis 1: Log Returns using Koopman DMD
    koopman_dmd_analysis(returns, axes[0], f"{ticker} Log Returns", "Log Return", delay_dim=10, rank=5)
    
    # Analysis 2: Actual Prices using Koopman DMD
    koopman_dmd_analysis(close_prices, axes[1], f"{ticker} Actual Prices", "Price (USD)", delay_dim=10, rank=5)

    plt.tight_layout()
    output_img = 'koopman_phase_portrait.png'
    plt.savefig(output_img, dpi=300)
    print(f"\nPlot saved successfully as '{os.path.abspath(output_img)}'")

if __name__ == "__main__":
    main()
