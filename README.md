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

# Proposed Solution (this is a Proof of Concept so far)

The current solution implemented draws heavily from [rufflewind](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation) and Simon Sappon fantastic [post](https://exyr.org/2018/rust-arenas-vs-dropck/) about Rust arena allocators.

The solution consists on separating the Computation Graph (or record) from the Tensors itself. This way, the Tensors don't need to live as long as the graph itself or hold mutable references to each other. 

The Computation Graph (CG) stores all the necessary information about how to calculate the gradients backwards from any given Tensor. Tensors are identified in the CG index. 

Let's check an example:


```Rust

pub fn main() {

    // Tensors only exist inside a graph, so here we store the Tensor data itself
    let mut parameter_store: HashMap<String, NdArray> = HashMap::new();


    for _i in 0..5 {
        // Where all the data to compute the gradients is stored
        let rec: ComputationRecord<NdArray> = ComputationRecord::new();
        // Here we create a linear layer with 3 input neurons and 3 output neurons.
        // This layer loads its parameters from the parameter_store or initializes them
        // randomly if they are not found there. In the future we will also pass an ID
        // to identify the layer uniquely along with its parameters, but for now its just
        // a PoC.
        let mut linear = LinearLayer::new(&rec, 3, 3, &parameter_store);

        // Create input data
        let mut data = NdArray::from_slice(&[1., 2., 3.]); // [3]
        data.reshape(&[1, 3]); // [1x3]

        // Create a tensor in the graph using this data
        let input = rec.tensor_from_value(data);

        // Run the input through the linear layer
        let linear_layer_output = linear.forward(&input);
        // Apply a relu
        let relu_output = relu(&linear_layer_output);
        // Sum all elements reducing them to shape [1]
        let loss = sum(&relu_output);
        println!("Loss: {:?}", loss.data());

        // get all the gradient from the loss tensor backwards
        let grad = loss.grad();
        let relu_grad = grad.wrt(&linear_layer_output);
        println!("Relu grad: {:?}", relu_grad.data());
        println!();
        // Optimize the linear layer in order to minimize the loss
        linear.optimize(grad, &mut parameter_store);
    }
}
```

Sample output:

```
Loss: NdArray([90.61549], shape=[1], strides=[1], layout=C | F (0x3), dynamic ndim=1)
Relu grad: NdArray([[1.0, 1.0, 1.0]], shape=[1, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2)

Loss: NdArray([90.19549], shape=[1], strides=[1], layout=C | F (0x3), dynamic ndim=1)
Relu grad: NdArray([[1.0, 1.0, 1.0]], shape=[1, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2)

Loss: NdArray([89.77549], shape=[1], strides=[1], layout=C | F (0x3), dynamic ndim=1)
Relu grad: NdArray([[1.0, 1.0, 1.0]], shape=[1, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2)

Loss: NdArray([89.35549], shape=[1], strides=[1], layout=C | F (0x3), dynamic ndim=1)
Relu grad: NdArray([[1.0, 1.0, 1.0]], shape=[1, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2)

Loss: NdArray([88.935486], shape=[1], strides=[1], layout=C | F (0x3), dynamic ndim=1)
Relu grad: NdArray([[1.0, 1.0, 1.0]], shape=[1, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2)
```



