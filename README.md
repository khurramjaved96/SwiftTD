# SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning

SwiftTD is an algorithm for learning value functions. It combines the ideas of step-size adaptation with the idea of a bound on the rate of learning. The implementations in this repository use linear function approximation. 

## Installation

```bash
pip install SwiftTD
```

## Usage

After installation, you can use the three implementations of SwiftTD in Python as: 

```python
import swifttd

# Version of SwiftTD that expects the full feature vector as input. This should only be used if the feature representation is not sparse. Otherwise, the sparse versions are more efficient.
td_dense = swifttd.SwiftTDNonSparse(
    num_features=5,     # Number of input features
    lambda_=0.95,        # Lambda parameter for eligibility traces
    initial_alpha=1e-2,  # Initial learning rate
    gamma=0.99,        # Discount factor
    eps=1e-5,          # Small constant for numerical stability
    max_step_size=0.1, # Maximum allowed step size
    step_size_decay=0.999, # Step size decay rate
    meta_step_size=1e-3,  # Meta learning rate
    eta_min=1e-10 # Minimum value of the step-size parameter
)

# Feature vector
features = [1.0, 0.0, 0.5, 0.2, 0.0] 
reward = 1.0
prediction = td_dense.step(features, reward)
print("Dense prediction:", prediction)

# Version of SwiftTD that expects the feature indices as input. This version assumes that the features are binary---0 or 1. For learning, the indices of the features that are 1 are provided. 
td_sparse = swifttd.SwiftTDBinaryFeatures(
    num_features=1000,     # Number of input features
    lambda_=0.95,        # Lambda parameter for eligibility traces
    initial_alpha=1e-2,  # Initial learning rate
    gamma=0.99,        # Discount factor
    eps=1e-5,          # Small constant for numerical stability
    max_step_size=0.1, # Maximum allowed step size
    step_size_decay=0.999, # Step size decay rate
    meta_step_size=1e-3,  # Meta learning rate
    eta_min=1e-10 # Minimum value of the step-size parameter
)

# Specify the indices of the features that are 1.
active_features = [1, 42, 999]  # Indices of active features
reward = 1.0
prediction = td_sparse.step(active_features, reward)
print("Sparse binary prediction:", prediction)

# Version of SwiftTD that expects the feature indices and values as input. This version does not assume that the features are binary. For learning, it expects a list of (index, value) pairs. Only the indices of the features that are non-zero need to be provided. 

td_sparse_nonbinary = swifttd.SwiftTD(
    num_features=1000,     # Number of input features
    lambda_=0.95,        # Lambda parameter for eligibility traces
    initial_alpha=1e-2,  # Initial learning rate
    gamma=0.99,        # Discount factor
    eps=1e-5,          # Small constant for numerical stability
    max_step_size=0.1, # Maximum allowed step size
    step_size_decay=0.999, # Step size decay rate
    meta_step_size=1e-3,  # Meta learning rate
    eta_min=1e-10 # Minimum value of the step-size parameter
)

# Specify the indices and values of the features that are non-zero.
feature_values = [(1, 0.8), (42, 0.3), (999, 1.2)]  # (index, value) pairs
reward = 1.0
prediction = td_sparse_nonbinary.step(feature_values, reward)
print("Sparse non-binary prediction:", prediction)
```

## Resources
- [Paper (PDF)](https://khurramjaved.com/swifttd.pdf)
- [Interactive Demo](https://khurramjaved.com/swifttd.html)

