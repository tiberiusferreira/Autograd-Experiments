use crate::TrackedTensor;
use crate::tensor_backends::{NdArray, TensorBackend};
use crate::tape::{ComputationRecord, Grad};
use crate::ops::*;
use std::collections::HashMap;


/// The input is 1xIN  we multiply by the weights INxOUT and get the output 1xOUT
pub struct LinearLayer<'a, T: TensorBackend>{
    // has shape INxOUT
    weights: TrackedTensor<'a, T>,
    id: String,
}

impl <'a, T: TensorBackend> LinearLayer<'a, T>{
    pub fn new(record: &'a ComputationRecord<T>, in_size: usize, out_size: usize, params_store: &HashMap<String, T>) -> Self{
        let id = "a".to_string();
        let param = match params_store.get(&id){
            None => {
                T::rand(&[in_size, out_size])
            },
            Some(val) => {
                val.clone()
            },
        };
        LinearLayer{
            weights: record.tensor_from_value(param),
            id
        }
    }
    /// Input must be of shape [1 x in_size]
    /// Output is of size [1 x out_size]
    pub fn forward(&mut self, input: &TrackedTensor<'a, T>) -> TrackedTensor<'a, T>{
        matmul(input, &self.weights)
    }

    pub fn optimize(&mut self, grad: Grad<T>, params_store: &mut HashMap<String, T>){
        let self_grads = grad.wrt(&self.weights);
        let updated_weights = self.weights.data().sub(&self_grads.data().mul_scalar(0.01));
        self.weights = self.weights.tape.tensor_from_value(updated_weights);
        params_store.insert(self.id.clone(), self.weights.data().clone());
    }
}


#[cfg(test)]
mod layer_tests {
    use crate::tape::ComputationRecord;
    use crate::tensor_backends::{NdArray, TensorBackend};
    use crate::ops::*;
    use crate::layers::LinearLayer;
    use std::collections::HashMap;

    #[test]
    fn matmul_test() {

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
}
