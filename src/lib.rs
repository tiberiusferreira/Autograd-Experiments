mod ops_impls;
mod store;
pub use ops_impls::*;
use std::collections::HashMap;
pub use store::ParameterStore;
pub mod test_helpers;
use ndarray::prelude::arr1;
use ndarray::{IxDyn, arr2, Axis};

pub trait Op: std::fmt::Debug {
    fn name(&self) -> String;
    /// Should return the result of the operation.
    /// The Tensor returned should have this Op as its "mother op"
    fn forward(self) -> Tensor;
    /// Should set its own operands gradients
    fn set_operand_grad(&mut self, previous_op_grad: ndarray::Array<f32, IxDyn>);
    fn operands(&self) -> Vec<&Tensor>;
    fn operands_mut(&mut self) -> Vec<&mut Tensor>;
    fn operands_shallow_clone(&self) -> Vec<Tensor>;
}

//
//#[test]
//pub fn test(){
//    let a = ndarray::arr1(&[2, 1]);
//    let b = ndarray::arr1(&[2, 1]);
////    let a = a.into_shape((1, 2)).unwrap();
////    a.do
////    let a = a.into_dyn();
////    let b = b.into_dyn();
//    use ndarray::Ix1;
//    use ndarray::Ix2;
//    let w = a.into_dimensionality::<Ix2>().unwrap();
////    println!("{}", w);
////    let k = a * b;
////    use ndarray::Ix1;
////    let d: Array1<OwnedRepr<f32>> = c.try_into().unwrap();
//}

#[derive(Debug)]
pub struct Tensor {
    pub data: ndarray::Array<f32, IxDyn>,
    shape: Vec<usize>,
    parameter_id: Option<String>,
    pub grad: Option<ndarray::Array<f32, IxDyn>>,
    mother_op: Option<Box<dyn Op>>,
}

impl Tensor {
    pub fn new(val: &[f32]) -> Self {
        /*/// let a = arr2(&[[1, 2, 3],
///                [4, 5, 6]]);*/
//        let mut v = vec![];
//        v.push(val.to_vec().as_slice());
//        let w = [val.to_vec()];
//        let k: [&[f32]; 1] = v.as_slice();

        let k = arr1(val);
//        let k = k.insert_axis(Axis(1));
        Tensor {
            data: k.into_dyn(),
            mother_op: None,
            parameter_id: None,
            grad: None,
            shape: vec![val.len()],
        }
    }

    pub fn from_ndarray(arr: ndarray::Array<f32, IxDyn>) -> Self {
        let shape = arr.shape().to_vec();
        Tensor {
            data: arr,
            mother_op: None,
            parameter_id: None,
            grad: None,
            shape,
        }
    }

    pub fn new_trainable(val: &[f32], id: String) -> Self {
        Tensor {
            data: arr1(val).into_dyn(),
            mother_op: None,
            parameter_id: Some(id),
            grad: None,
            shape: vec![val.len()],
        }
    }

    pub fn shallow_clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            mother_op: None,
            parameter_id: self.parameter_id.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
        }
    }

    pub fn backwards(&mut self, initial_grad: Option<ndarray::Array<f32, IxDyn>>) -> ParameterStore {
        if let Some(grad) = initial_grad{
            self.grad = Some(grad);
        }
        // where the parameters will be stored after having the gradients populated
        let mut hash: HashMap<String, Tensor> = HashMap::new();

        // for each operation, calculate operands gradients recursively
        self.recursive_calc_grads(&mut hash);

        ParameterStore::from_hashmap(hash)
    }

    pub fn recursive_calc_grads(&mut self, hash: &mut HashMap<String, Tensor>) {
        if let Some(op) = &mut self.mother_op {
            // Set the gradient of this tensor's original Op arguments
            let one = ndarray::arr1(&[1.]).into_dyn();
            op.set_operand_grad(self.grad.clone().unwrap_or(one));

            // Ask the operands to set their Ops operands gradients too
            for operand in op.operands_mut() {
                operand.recursive_calc_grads(hash);
            }

            // Now all gradients should have been updated
            let operands = op.operands();

            // Get all tensors which are parameters
            let trainable_operands: Vec<Tensor> = operands
                .iter()
                .filter(|o| o.parameter_id.is_some())
                .map(|o| o.shallow_clone())
                .collect();

            // Insert parameters in the Hashmap
            for single_trainable in trainable_operands {
                let id = single_trainable.parameter_id.clone().unwrap();
                hash.insert(id, single_trainable);
            }
        }
    }
}
