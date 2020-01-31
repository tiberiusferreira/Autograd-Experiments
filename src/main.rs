use backprop::*;
use ndarray::arr2;

fn main() {
    let mut vs: ParameterStore = ParameterStore::new();

    //    for _i in 0..1 {
//    let a = vs.remove_or_init("a".into(), &[-3., -2.]);
    let a = arr2(&[[1., 2., 3.],
        [4., 5., 6.]]).into_dyn();
    let a = Tensor::from_ndarray(a);
    let b = Tensor::new(&[2., -1., 5.]);

    let c = a*b;
    println!("{:?}", c.data);
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
