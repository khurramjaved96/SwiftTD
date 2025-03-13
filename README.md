# SwiftTD

SwiftTD is an implementation of temporal difference learning with adaptive step sizes. It can be combined with neural networks by applying it to only the last layer.

## Usage
This repository has two implementations for temporal difference learning:

1. **SwiftTDDense**: This implementation uses dense feature vectors, making it suitable for scenarios where all features are available and relevant. It is designed to work efficiently with continuous data.

2. **SwiftTDSparse**: This implementation is optimized for sparse feature vectors, where only a subset of features are active or relevant at any given time. It is ideal for situations with large feature spaces and sparse data representation.

Both implementations provide a `Step()` method for updating the value function:

- `SwiftTDDense::Step(std::vector<float> &features, float reward)`
- `SwiftTDSparse::Step(std::vector<int> &feature_indices, float reward)`

The main difference is that `SwiftTDSparse` takes feature indices instead of dense feature values.

## Resources
- [Paper (PDF)](https://khurramjaved.com/swifttd.pdf)
- [Interactive Demo](https://khurramjaved.com/swifttd.html)

