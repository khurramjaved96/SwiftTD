# SwiftTD (Python and C++)

SwiftTD is an implementation of temporal difference learning with adaptive step sizes. It can be combined with neural networks by applying it to only the last layer.

## Installation

### PyPI Installation (Recommended)

The easiest way to install SwiftTD is via PyPI:

```bash
pip install swifttd
```


## Usage

### C++ Usage
This repository has three implementations for temporal difference learning:

1. **SwiftTDDense**: This implementation uses dense feature vectors, making it suitable for scenarios where all features are available and relevant. It is designed to work efficiently with continuous data.

2. **SwiftTDSparse**: This implementation is optimized for sparse feature vectors with binary features, where only a subset of features are active or relevant at any given time. It is ideal for situations with large feature spaces and sparse data representation.

3. **SwiftTDSparseAndNonBinaryFeatures**: This implementation handles sparse feature vectors with non-binary values, allowing for weighted features where each feature can have a specific value rather than just being present or absent.

All implementations provide a `Step()` method for updating the value function and returning the value prediction:

- `float prediction = SwiftTDDense::Step(std::vector<float> &features, float reward)`
- `float prediction = SwiftTDSparse::Step(std::vector<int> &feature_indices, float reward)`
- `float prediction = SwiftTDSparseAndNonBinaryFeatures::Step(std::vector<std::pair<int, float>> &feature_indices_values, float reward)`

The main differences between the implementations are:
- `SwiftTDDense` takes dense feature vectors
- `SwiftTDSparse` takes feature indices (binary features)
- `SwiftTDSparseAndNonBinaryFeatures` takes feature index-value pairs (weighted features)

### Python Usage

After installation, you can use SwiftTD in Python:

```python
import swifttd

# Create a dense TD learner
td_dense = swifttd.SwiftTDDense(
    num_features=5,     # Number of input features
    lambda_=0.95,        # Lambda parameter for eligibility traces
    initial_alpha=1e-2,  # Initial learning rate
    gamma=0.99,        # Discount factor
    eps=1e-8,          # Small constant for numerical stability
    max_step_size=0.1, # Maximum allowed step size
    step_size_decay=0.99, # Step size decay rate
    meta_step_size=1e-3  # Meta learning rate
)

# Use dense features
features = [1.0, 0.0, 0.5, 0.2, 0.0]  # Dense feature vector
reward = 1.0
prediction = td_dense.step(features, reward)
print("Dense prediction:", prediction)

# Create a sparse binary TD learner
td_sparse = swifttd.SwiftTDSparse(
    num_features=1000,  # Can handle larger feature spaces efficiently
    lambda_=0.95,
    initial_alpha=1e-2,
    gamma=0.99,
    eps=1e-8,
    max_step_size=0.1,
    step_size_decay=0.99,
    meta_step_size=1e-3
)

# Use sparse binary features (only active feature indices)
active_features = [1, 42, 999]  # Indices of active features
reward = 1.0
prediction = td_sparse.step(active_features, reward)
print("Sparse binary prediction:", prediction)

# Create a sparse non-binary TD learner
td_sparse_nonbinary = swifttd.SwiftTDSparseNonBinary(
    num_features=1000,  # Can handle larger feature spaces efficiently
    lambda_=0.95,
    initial_alpha=1e-2,
    gamma=0.99,
    eps=1e-8,
    max_step_size=0.1,
    step_size_decay=0.99,
    meta_step_size=1e-3
)

# Use sparse non-binary features (feature index-value pairs)
feature_values = [(1, 0.8), (42, 0.3), (999, 1.2)]  # (index, value) pairs
reward = 1.0
prediction = td_sparse_nonbinary.step(feature_values, reward)
print("Sparse non-binary prediction:", prediction)
```

## Resources
- [Paper (PDF)](https://khurramjaved.com/swifttd.pdf)
- [Interactive Demo](https://khurramjaved.com/swifttd.html)

