# Autograd-Experiments

Experiments in automatic differentiation. The idea is to imagine PyTorch, but in written in and for Rust. Nothing serious, for fun!


## Current / Future Work: 

- [ ] Add / Sub / Mul should work with both Scalar and Rank 2 Tensors, for now only works with Rank 2. 
- [ ] Support creation syntax closer to Pytorch. Something like: ```let x = Tensor::from(&[[2., 3.], [2., 3.]])``` currently nested slices are not supported, only this: ```Tensor::from(&[2., 3.])```. Investigate if possible.
