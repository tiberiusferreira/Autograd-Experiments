# Autograd-Experiments

Experiments in automatic differentiation. The idea is to imagine PyTorch, but in written in and for Rust. Nothing serious, for fun!


# Design Problem

The first design tried to copy as close as possible Pytorch's API from the user perspective. However, consider the following code: 

```Python
import torch;

x = torch.tensor([2], requires_grad=True, dtype=torch.float)
y = torch.tensor([3], requires_grad=True, dtype=torch.float)

for k in range(2):
  z = (x*y)
  z.backward() # fills x and y gradients, so z must have a mutable reference of some kind to x and y
  print('x_grad =', x.grad)
  x.data = x.data-0.1*x.grad # here we are modifying x in-place, while z holds the mutable reference
  x.grad.data.zero_() 
  print('x =', x)

print('z =', z)
```
Which prints: 
```Python
x_grad = tensor([3.])
x = tensor([1.7000], requires_grad=True)
x_grad = tensor([3.])
x = tensor([1.4000], requires_grad=True)
z = tensor([5.1000], grad_fn=<MulBackward0>)
```

As one can see, in the code above Z holds mutable access to X and Y and X modifies itself in place while this mutable access is still valid. 

Rust disallow it for [good reasons](https://manishearth.github.io/blog/2015/05/17/the-problem-with-shared-mutability/). One *could* implement it in Rust using [Rc<RefCell<>>](https://doc.rust-lang.org/book/ch15-05-interior-mutability.html).

But this just sidesteps the root problem here: it is not clear who owns x and y at any given time. 

# Solution

Blog post coming up soon!






## Current / Future Work: 

- [ ] Add / Sub / Mul should work with both Scalar and Rank 2 Tensors, for now only works with Rank 2. 
- [ ] Support creation syntax closer to Pytorch. Something like: ```let x = Tensor::from(&[[2., 3.], [2., 3.]])``` currently nested slices are not supported, only this: ```Tensor::from(&[2., 3.])```. Investigate if possible.

