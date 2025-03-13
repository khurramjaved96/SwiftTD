# SwiftTD (Python and C++)

SwiftTD is an implementation of temporal difference learning with adaptive step sizes. It can be combined with neural networks by applying it to only the last layer.

## Installation

First, install pybind11 which is required for Python bindings:

```bash
# On Ubuntu/Debian
sudo apt-get install pybind11-dev

# On macOS with Homebrew
brew install pybind11

# On Windows with vcpkg
vcpkg install pybind11

# Or via pip (works on all platforms)
pip install pybind11
```

### Building and Installing

```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

## Usage

### C++ Usage
This repository has two implementations for temporal difference learning:

1. **SwiftTDDense**: This implementation uses dense feature vectors, making it suitable for scenarios where all features are available and relevant. It is designed to work efficiently with continuous data.

2. **SwiftTDSparse**: This implementation is optimized for sparse feature vectors, where only a subset of features are active or relevant at any given time. It is ideal for situations with large feature spaces and sparse data representation.

Both implementations provide a `Step()` method for updating the value function and returning the value prediction:

- `float prediction = SwiftTDDense::Step(std::vector<float> &features, float reward)`
- `float prediction = SwiftTDSparse::Step(std::vector<int> &feature_indices, float reward)`

The main difference between the two implementations is that `SwiftTDSparse` takes feature indices instead of dense feature values.

### Python Usage

After installation, you can use SwiftTD in Python:

```python
import swift_td

# Create a dense TD learner
td_dense = swift_td.SwiftTDDense(
    num_features=5,     # Number of input features
    lambda_=0.99,        # Lambda parameter for eligibility traces
    initial_alpha=1e-6,  # Initial learning rate
    gamma=0.99,        # Discount factor
    eps=1e-8,          # Small constant for numerical stability
    max_step_size=0.5, # Maximum allowed step size
    step_size_decay=0.99, # Step size decay rate
    meta_step_size=1e-3  # Meta learning rate
)

# Use dense features
features = [1.0, 0.0, 0.5, 0.2, 0.0]  # Dense feature vector
reward = 1.0
prediction = td_dense.step(features, reward)

# Get learned weights
weights = td_dense.get_weights()

# Create a sparse TD learner
td_sparse = swift_td.SwiftTDSparse(
    num_features=1000,  # Can handle larger feature spaces efficiently
    lambda_=0.99,
    initial_alpha=1e-6,
    gamma=0.99,
    eps=1e-8,
    max_step_size=0.5,
    step_size_decay=0.99,
    meta_step_size=1e-3
)

# Use sparse features (only active feature indices)
active_features = [1, 42, 999]  # Indices of active features
reward = 1.0
prediction = td_sparse.step(active_features, reward)
```

## Resources
- [Paper (PDF)](https://khurramjaved.com/swifttd.pdf)
- [Interactive Demo](https://khurramjaved.com/swifttd.html)

