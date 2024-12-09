import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch and preprocess historical stock data
tickers = ['PLTR', 'MSFT', 'GOOGL', 'AMZN', 'SHEL.L']
data = yf.download(tickers, start="2015-01-01", end="2024-12-31")['Adj Close']

# Fill missing data using forward-fill method to maintain consistency in time series
data.fillna(method='ffill', inplace=True)

# Step 2: Calculate daily returns of the stocks
returns = data.pct_change().dropna()

# Standardise the returns for comparability across different assets
normalised_returns = (returns - returns.mean()) / returns.std()

# Function to perform Principal Component Analysis (PCA)
def perform_pca(data):
    """
    Conduct Principal Component Analysis (PCA) on the given data.
    This function computes the covariance matrix, eigenvalues, eigenvectors,
    and the explained variance ratio for the input data.
    
    Args:
    data (pd.DataFrame): Normalised returns of the assets
    
    Returns:
    tuple: Eigenvalues, eigenvectors, and explained variance ratio
    """
    # Compute the covariance matrix of the normalised returns
    cov_matrix = np.cov(data.T)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the proportion of variance explained by each principal component
    total_variance = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_variance

    return eigenvalues, eigenvectors, explained_variance_ratio

# Perform PCA on the normalised returns
eigenvalues, eigenvectors, explained_variance_ratio = perform_pca(normalised_returns)

# Create a DataFrame to represent eigenportfolios (i.e., the principal components)
eigenportfolios = pd.DataFrame(
    eigenvectors,
    index=tickers,
    columns=[f"PC{i+1}" for i in range(len(tickers))]
)

# Display the explained variance ratio for each principal component
print("Explained Variance Ratio by Principal Component:")
for i, var in enumerate(explained_variance_ratio, start=1):
    print(f"PC{i}: {var:.2%}")

# Step 3: Backtest the eigenportfolios by computing cumulative returns
def backtest(weights, returns):
    """
    Backtest a portfolio by computing cumulative returns based on the given weights.
    
    Args:
    weights (pd.Series): Portfolio weights corresponding to each asset
    returns (pd.DataFrame): Daily returns of the assets
    
    Returns:
    pd.Series: Cumulative returns of the portfolio
    """
    # Compute the weighted sum of asset returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Compute the cumulative product of portfolio returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns

# Backtest the first two eigenportfolios (PC1 and PC2)
cumulative_returns_pc1 = backtest(eigenportfolios['PC1'], returns)
cumulative_returns_pc2 = backtest(eigenportfolios['PC2'], returns)

# Step 4: Visualise the results through graphical representations
# Plot the cumulative returns of the first two eigenportfolios
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns_pc1, label="PC1 Portfolio")
plt.plot(cumulative_returns_pc2, label="PC2 Portfolio")
plt.title("Cumulative Returns of Eigenportfolios (PC1 and PC2)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.show()

# Plot the explained variance ratio for each principal component
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(eigenvalues) + 1), explained_variance_ratio * 100, color='skyblue')
plt.title("Explained Variance Ratio by Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.grid(True)
plt.show()
