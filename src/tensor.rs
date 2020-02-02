use crate::{Op, ParameterStore};
use ndarray::arr1;
use ndarray::prelude::IxDyn;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Tensor {
    pub data: ndarray::Array<f32, IxDyn>,
    pub(in crate) shape: Vec<usize>,
    pub(in crate) parameter_id: Option<String>,
    pub grad: Option<ndarray::Array<f32, IxDyn>>,
    pub(in crate) mother_op: Option<Box<dyn Op>>,
}

impl Tensor {
    pub fn new(val: &[f32]) -> Self {
        Tensor {
            data: arr1(val).into_dyn(),
            mother_op: None,
            parameter_id: None,
            grad: None,
            shape: vec![val.len()],
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let val = ndarray::prelude::ArrayBase::zeros(shape);
        Tensor {
            data: val.into_dyn(),
            mother_op: None,
            parameter_id: None,
            grad: None,
            shape: shape.to_vec(),
        }
    }

    pub fn rand(shape: &[usize]) -> Self {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let val = ndarray::Array::random(shape, Uniform::new(0., 10.));
        Tensor {
            data: val.into_dyn(),
            mother_op: None,
            parameter_id: None,
            grad: None,
            shape: shape.to_vec(),
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

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn enable_training(&mut self, id: String) {
        self.parameter_id = Some(id);
    }

    pub fn disable_training(&mut self) {
        self.parameter_id = None;
    }

    pub fn clone_without_op_graph(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            mother_op: None,
            parameter_id: self.parameter_id.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
        }
    }

    pub fn backwards(&mut self, initial_grad: Option<Tensor>) -> ParameterStore {
        if let Some(grad) = initial_grad {
            assert_eq!(
                self.shape, grad.shape,
                "Gradient needs to have the same shape as the tensor itself"
            );
            self.grad = Some(grad.data);
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
                .map(|o| o.clone_without_op_graph())
                .collect();

            // Insert parameters in the Hashmap
            for single_trainable in trainable_operands {
                let id = single_trainable.parameter_id.clone().unwrap();
                hash.insert(id, single_trainable);
            }
        }
    }
}
