mod ops_impls;
mod store;
pub use store::ParameterStore;
use std::collections::HashMap;
pub use ops_impls::*;

pub trait Op: std::fmt::Debug{
    //std::fmt::Debug + std::clone::Clone + Sized
    fn name(&self) -> String;
    fn forward(self) -> Tensor;
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

    pub fn shallow_clone(&self) -> Self {
        Tensor {
            data: self.data,
            mother_op: None,
            parameter_id: self.parameter_id.clone(),
            grad: self.grad.clone()
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

    pub fn get_trainable_with_grads(&mut self) -> ParameterStore{
        // where the parameters will be stored after having the gradients populated
        let mut hash: HashMap<String, Tensor> = HashMap::new();

        // for each operation, calculate operands gradients recursively
        self.recursive_calc_grads(&mut hash);


        ParameterStore::from_hashmap(hash)

    }

    pub fn recursive_calc_grads(&mut self, hash: &mut HashMap<String, Tensor>){
        if let Some(op) = &mut self.mother_op {
            op.set_operand_grad(self.grad.unwrap_or(1.));

            for operand in op.operands_mut(){
                operand.recursive_calc_grads(hash);
            }

            let operands = op.operands();

            let trainable_operands: Vec<Tensor> = operands.iter()
                .filter(|o| o.parameter_id.is_some())
                .map(|o| o.shallow_clone())
                .collect();

            for single_trainable in trainable_operands {
                let id = single_trainable.parameter_id.clone().unwrap();
                hash.insert(id, single_trainable);
            }

        }

    }

}
