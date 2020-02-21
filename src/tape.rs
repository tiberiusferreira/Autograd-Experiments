use std::cell::RefCell;
use std::fmt::{Error, Formatter};
use crate::tensor_backends::TensorBackend;


#[derive(Debug)]
pub struct Tape<T: TensorBackend> {
    /// Stores the information necessary to calculate the gradient of the operands of Tensors
    /// which reference this Tape. Each Tensor stores an Index into this structure.
    /// So, in order to construct the gradients we start with a Tensor and its given gradient,
    /// typically 1 if it stores a single value.
    /// Then we calculate its parents gradients using the data stored in its node, then for each
    /// of those parents we calculate their gradient and so on.
    ops_data: RefCell<Vec<OpData<T>>>,
}

#[derive(Debug)]
pub struct OpData<T: TensorBackend> {
    /// Stores the data necessary to calculate the gradient of each of the parents (operands) of
    /// the Op which created this Tensor. If the Var was user create, this is empty.
    pub operands_grad_blueprint: Vec<OperandGradBlueprint<T>>,
    pub op_name: String,
}

impl <T: TensorBackend> OpData<T> {
    pub fn empty() -> Self {
        Self {
            operands_grad_blueprint: vec![],
            op_name: "NoOp".to_string(),
        }
    }

    pub fn from_blueprints(blueprints: Vec<OperandGradBlueprint<T>>, op_name: String) -> Self {
        Self {
            operands_grad_blueprint: blueprints,
            op_name,
        }
    }
}

/// First argument is the child_grad, second is the current "parent" grad
pub struct GradFn<T: TensorBackend>(pub Box<dyn Fn(T, &mut T)>);

impl <T: TensorBackend> std::fmt::Debug for GradFn<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_str("GradFn")
    }
}

#[derive(Debug)]
pub struct OperandGradBlueprint<T: TensorBackend> {
    /// The index where to store the gradient calculated by the grad_fn in the output gradient
    /// structure. The gradient calculated is the gradient of the operand in question
    /// the Var associated with this struct.
    /// This index is the same as the one inside the parent Var.
    operand_tape_index: usize,
    /// Used to initialize the gradients (to zero)
    grad_shape: Vec<usize>,
    /// Function which takes the current Var gradient and a mutable reference to the current gradient
    /// of one of the operands of the operation that resulted in this Var
    grad_fn: GradFn<T>,
}

#[derive(Debug)]
pub struct Tensor<'t, T: TensorBackend> {
    /// Reference to the Tape which stores the information needed to calculate the gradients
    pub tape: &'t Tape<T>,
    /// Index of the slot in the tape where the information to calculate the gradient of the
    /// "parents" of this Var are stored
    pub parent_op_index: usize,
    /// The actual value of this Var
    pub value: T,
}

impl <T: TensorBackend> Tape<T> {
    pub fn new() -> Self {
        Tape {
            ops_data: RefCell::new(Vec::new()),
        }
    }

    //noinspection RsNeedlessLifetimes
    pub fn new_tensor<'t>(&'t self, value: &[f32]) -> Tensor<'t, T> {
        Tensor {
            tape: self,
            value: T::from_slice(value),
            parent_op_index: self.push_op(OpData::empty()),
        }
    }

    pub fn len(&self) -> usize {
        self.ops_data.borrow().len()
    }

    pub fn push_op(&self, op_data: OpData<T>) -> usize {
        let mut ops_data = self.ops_data.borrow_mut();
        let len = ops_data.len();
        ops_data.push(op_data);
        len
    }
}

// pub fn mul_vec_f64(left: &Vec<f64>, right: &Vec<f64>) -> Vec<f64> {
//     assert_eq!(left.len(), right.len(), "right and left lens not equal");
//     let len = left.len();
//     let mut result = Vec::with_capacity(len);
//     for i in 0..left.len() {
//         result.push(left[i] * right[i]);
//     }
//     result
// }

#[derive(Debug)]
pub struct Grad<T: TensorBackend> {
    all_grads: Vec<T>,
}

impl <T: TensorBackend> Grad<T> {
    //noinspection RsNeedlessLifetimes
    pub fn wrt<'t>(&self, var: &Tensor<'t, T>) -> T {
        match self.all_grads.get(var.parent_op_index) {
            None => {
                panic!("This var is not part of the computational graph. Maybe it was created using another Tape");
            }
            Some(grad) => grad.clone(),
        }
    }
}

impl<'t, T: TensorBackend> Tensor<'t, T> {
    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn grad(&self) -> Grad<T> {
        let tape_len = self.tape.len();
        let ops_data = self.tape.ops_data.borrow();

        let mut all_grads: Vec<T> = vec![T::empty(); tape_len];
        // Set self gradient as 1.0
        let one = T::from_slice(&[1.]);
        all_grads[self.parent_op_index] = one;

        let mut tape_indices_to_visit = vec![self.parent_op_index];

        while let Some(current_tape_index) = tape_indices_to_visit.pop() {
            // Get the data to calculate current Var parents gradients
            let current_op_data = &ops_data[current_tape_index];
            // Get current Var gradient
            let current_tensor_grad = all_grads[current_tape_index].clone();
            // For each parent of this Var
            for operand in &current_op_data.operands_grad_blueprint {
                // Make sure to visit parents later
                tape_indices_to_visit.push(operand.operand_tape_index);
                // Get the function to calculate the gradient
                let grad_fn = &operand.grad_fn;
                // Get the current gradient
                let curr_grad = &mut all_grads[operand.operand_tape_index];
                // If empty, initialize it
                if curr_grad.is_empty() {
                    let new_grad_shape = operand.grad_shape.clone();
                    *curr_grad = T::zeros(new_grad_shape.as_slice());
                }
                // Update its gradient
                grad_fn.0(current_tensor_grad.clone(), curr_grad);
            }
        }

        Grad { all_grads }
    }

    /// Returns a blueprint to calculate this Var's gradient using the provided grad_fn
    pub fn self_gradient_blueprint(&self, grad_fn: GradFn<T>) -> OperandGradBlueprint<T> {
        OperandGradBlueprint {
            operand_tape_index: self.parent_op_index,
            grad_shape: self.value.shape().to_vec(),
            grad_fn,
        }
    }


}

