import yfinance as yf
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Fetch stock data from Yahoo Finance
print("\n\nEnter Yahoo Finance Ticker:\n\n")
ticker = input()
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")['Adj Close']

# Calculate daily returns (percentage change)
returns = data.pct_change().dropna()

# Discretise returns into custom states based on thresholds
def discretize_returns(returns):
    states = []
    for ret in returns:
        if ret > 0.02:  # High Positive (2% and above)
            states.append('High Positive')
        elif ret > 0.005:  # Moderate Positive (0.5% - 2%)
            states.append('Moderate Positive')
        elif ret > -0.005:  # No Change (-0.5% to 0.5%)
            states.append('No Change')
        elif ret > -0.02:  # Moderate Negative (-2% to -0.5%)
            states.append('Moderate Negative')
        else:  # High Negative (-2% and below)
            states.append('High Negative')
    return states

# Apply discretisation to returns
states = discretize_returns(returns.values)

# Create the transition matrix based on state transitions
def transition_matrix(states):
    state_space = ['High Positive', 'Moderate Positive', 'No Change', 'Moderate Negative', 'High Negative']
    transition_counts = pd.DataFrame(np.zeros((5, 5)), columns=state_space, index=state_space)

    # Count transitions between states
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_counts.loc[current_state, next_state] += 1

    # Convert counts to probabilities
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    # Handle rows that sum to zero by assigning small probabilities
    transition_matrix = transition_matrix.fillna(1e-5)

    # Normalize to ensure each row sums to 1
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    
    return transition_matrix

# Generate the transition matrix
transition_matrix_df = transition_matrix(states)

# Visualise the transition matrix as a network graph
def plot_markov_chain(transition_matrix):
    G = nx.DiGraph()

    # Add nodes (states) to the graph
    for state in transition_matrix.index:
        G.add_node(state)

    # Add edges with transition probabilities as weights
    for from_state in transition_matrix.index:
        for to_state in transition_matrix.columns:
            probability = transition_matrix.loc[from_state, to_state]
            if probability > 0:
                G.add_edge(from_state, to_state, weight=probability)

    # Define the layout for the graph
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Markov Chain for Amazon Stock Price Movement")
    plt.show()

# Plot the network graph
plot_markov_chain(transition_matrix_df)
