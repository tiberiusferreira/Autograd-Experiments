use backprop::*;


fn main() {
    let mut vs: ParameterStore = ParameterStore::new();

    for _i in 0..1 {
        let a = vs.remove_or_init("a".into(), -3.);
        let b = Tensor::new(2.);
        let mut c = a * b;
        c = relu(c);
        println!("{:#?}", c);
        vs = c.get_trainable_with_grads();
        println!("{:#?}", c);
        for (_id, tensor) in &mut vs {
            tensor.data = tensor.data - tensor.grad.unwrap()*0.1;
            tensor.grad = None;
        }

//        println!("{:?}", c.data);
    }

}

