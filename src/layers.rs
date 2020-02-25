use crate::TrackedTensor;
use crate::tensor_backends::{NdArray, TensorBackend};
use crate::tape::{ComputationRecord, Grad};
use crate::ops::*;
use std::collections::HashMap;

enum LayerParam<'a, T: TensorBackend>{
    Tracked(TrackedTensor<'a, T>),
    Raw(T)
}
/// The input is 1xIN  we multiply by the weights INxOUT and get the output 1xOUT
pub struct LinearLayer<'a, T: TensorBackend>{
    // has shape INxOUT
    weights: LayerParam<'a, T>,
    id: String,
}

impl <'a, T: TensorBackend> LinearLayer<'a, T>{
    pub fn new(in_size: usize, out_size: usize, params_store: &HashMap<String, T>) -> Self{
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
            weights: LayerParam::Raw(param),
            id
        }
    }
    pub fn forward(&mut self, input: &TrackedTensor<'a, T>) -> TrackedTensor<'a, T>{

        let weights = match &mut self.weights {
            LayerParam::Tracked(_) => {
                panic!();
            },
            LayerParam::Raw(tensor) => {
                let mut raw_tensor = T::empty();
                std::mem::swap(&mut raw_tensor, tensor);
                raw_tensor
            },
        };
        self.weights = LayerParam::Tracked(input.tape.tensor_from_value(weights));
        match &self.weights{
            LayerParam::Tracked(weights) => {
                matmul(input, &weights)
            },
            LayerParam::Raw(_) => {
                panic!("")
            },
        }

    }

    pub fn optimize(&mut self, grad: Grad<T>, params_store: &mut HashMap<String, T>){
        match &mut self.weights {
            LayerParam::Tracked(tracked_param) => {
                // let self_grads = grad.wrt(&self.weights);
                // let updated_weights = self.weights.data().sub(&self_grads.data().mul_scalar(0.1));
                // self.weights = self.weights.tape.tensor_from_value(updated_weights);
                // params_store.insert(self.id.clone(), self.weights.data().clone());
            },
            LayerParam::Raw(_) => {
                panic!("Cant optimized before a call to forward");
            },
        }

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

        //
        let mut linear = LinearLayer::new(3, 5, &parameter_store);

        for _i in 0..100 {
            let t: ComputationRecord<NdArray> = ComputationRecord::new();


            let mut data = NdArray::from_slice(&[1., 2., 3.]); // 1x3
            data.reshape(&[1, 3]);

            let input = t.tensor_from_value(data);
            let output = linear.forward(&input);
            let loss = sum(&output);
            println!("{:?}", loss.data());
            //
            let grad = loss.grad();
            linear.optimize(grad, &mut parameter_store);
        }
        //


    }
}
