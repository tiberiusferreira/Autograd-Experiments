mod ops_impls;
mod store;
pub use store::ParameterStore;
use std::collections::HashMap;
pub use ops_impls::*;

pub trait Op: std::fmt::Debug{
    fn name(&self) -> String;
    /// Should return the result of the operation.
    /// The Tensor returned should have this Op as its "mother op"
    fn forward(self) -> Tensor;
    /// Should set its own operands gradients
    fn set_operand_grad(&mut self, previous_op_grad: f32);
    fn operands(&self) -> Vec<&Tensor>;
    fn operands_mut(&mut self) -> Vec<&mut Tensor>;
    fn operands_shallow_clone(&self) -> Vec<Tensor>;
}



#[derive(Debug)]
pub struct Tensor {
    pub data: f32,
    parameter_id: Option<String>,
    pub grad: Option<f32>,
    mother_op: Option<Box<dyn Op>>,
}

impl Tensor {
    pub fn new(val: f32) -> Self {
        Tensor {
            data: val,
            mother_op: None,
            parameter_id: None,
            grad: None
        }
    }

    pub fn new_trainable(val: f32, id: String) -> Self {
        Tensor {
            data: val,
            mother_op: None,
            parameter_id: Some(id),
            grad: None
        }
    }

    pub fn shallow_clone(&self) -> Self {
        Tensor {
            data: self.data,
            mother_op: None,
            parameter_id: self.parameter_id.clone(),
            grad: self.grad.clone()
        }
    }

    pub fn get_trainable_with_grads(&mut self) -> ParameterStore{
        // where the parameters will be stored after having the gradients populated
        let mut hash: HashMap<String, Tensor> = HashMap::new();

        // for each operation, calculate operands gradients recursively
        self.recursive_calc_grads(&mut hash);


        ParameterStore::from_hashmap(hash)

    }

    pub fn recursive_calc_grads(&mut self, hash: &mut HashMap<String, Tensor>){
        if let Some(op) = &mut self.mother_op {
            // Set the gradient of this tensor's original Op arguments
            op.set_operand_grad(self.grad.unwrap_or(1.));

            // Ask the operands to set their Ops operands gradients too
            for operand in op.operands_mut(){
                operand.recursive_calc_grads(hash);
            }

            // Now all gradients should have been updated
            let operands = op.operands();

            // Get all tensors which are parameters
            let trainable_operands: Vec<Tensor> = operands.iter()
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
