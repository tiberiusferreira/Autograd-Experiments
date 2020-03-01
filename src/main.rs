use backprop;
use std::collections::HashMap;
use backprop::tensor_backends::{NdArray, TensorBackend};
use backprop::layers::LinearLayer;
use backprop::ComputationRecord;
use backprop::ops::*;

pub fn main() {

    let mut parameter_store: HashMap<String, NdArray> = HashMap::new();

    for _i in 0..1000 {
        let rec: ComputationRecord<NdArray> = ComputationRecord::new();
        let mut linear = LinearLayer::new(&rec, 3, 3, &parameter_store);


        let mut data = NdArray::from_slice(&[1., 2., 3.]); // 1x3
        data.reshape(&[1, 3]);

        let input = rec.tensor_from_value(data);

        let output = linear.forward(&input);
        let output = relu(&output);
        let loss = sum(&output);
        println!("{:?}", loss.data());


        let grad = loss.grad();
        linear.optimize(grad, &mut parameter_store);
    }
}
