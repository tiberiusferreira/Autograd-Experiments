use backprop::*;
use ndarray::arr2;

fn main() {
    //    let mut vs: ParameterStore = ParameterStore::new();

    //    for _i in 0..1 {
    //    let a = vs.remove_or_init("a".into(), &[-3., -2.]);
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]).into_dyn(); // 2x3

    let a: Tensor<NdArray> = Tensor::from_ndarray(a);

//    let b = arr2(&[[7., 8., 9.]]).reversed_axes().into_dyn();
    // 3x1

//    let b = Tensor::from_ndarray(b);
//    let g = Tensor::from_ndarray(arr2(&[[1.], [1.]]).into_dyn());
    //    let b = Tensor::new();

//    let mut c = matmul(a, b); // 2x1
//    c.backwards(Some(g));
//    println!("{:#?}", c);
    //        let mut c = a * b;
    //        c = relu(c);
    //        println!("{:#?}", c);
    //        vs = c.backwards();
    //        println!("{:#?}", c);
    //        for (_id, tensor) in &mut vs {
    //            tensor.data = tensor.data - tensor.grad.unwrap()*0.1;
    //            tensor.grad = None;
    //        }
    //
    //        println!("{:?}", c.data);
    //    }
}
