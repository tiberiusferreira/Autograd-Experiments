use crate::{Op, ParameterStore};
use ndarray::arr1;
use ndarray::prelude::IxDyn;
use std::collections::HashMap;
pub use crate::tensor::backends::NdArray;

mod backends;
pub use backends::TensorBackend;


pub struct Grad<T: TensorBackend>(Cell<Option<Box<Tensor<T>>>>);

impl <T: TensorBackend> Debug for Grad<T>{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let k = self.0.take();
        let r = k.fmt(f);
        self.0.replace(k);
        r
    }
}

impl <T: TensorBackend> Grad<T>{
    pub fn none() -> Self{
        Grad(Cell::new(None))
    }

    pub fn set(&self, t: Tensor<T>){
        self.0.replace(Some(Box::new(t)));
    }

    pub fn clone(&self) -> Self{
        let k = self.0.take();
        let clone = k.as_ref().map(|e| Box::new(e.clone_only_data()));
        self.0.replace(k);
        Grad(Cell::new(clone))
    }

}
#[derive(Debug)]
pub struct Tensor<T: TensorBackend> {
    pub data: T,
    pub(in crate) parameter_id: Option<String>,
    pub grad: Grad<T>,
    pub(in crate) mother_op: Option<Box<dyn Op<T>>>,
}

impl Tensor<NdArray> {
    pub fn from_ndarray(arr: ndarray::Array<f32, IxDyn>) -> Tensor<NdArray> {
        Tensor {
            data: NdArray(arr),
            mother_op: None,
            parameter_id: None,
            grad: Grad::none(),
        }
    }
}

impl<T: TensorBackend> Tensor<T> {
    pub fn new(val: &[f32]) -> Self {
        Tensor {
            data: T::from_slice(val),
            mother_op: None,
            parameter_id: None,
            grad: Grad::none(),
        }
    }

    pub fn from_op_result(op_result: T, mother_op: Box<dyn Op<T>>) -> Self {
        Tensor {
            data: op_result,
            mother_op: Some(mother_op),
            parameter_id: None,
            grad: Grad::none(),
        }
    }

    pub (in crate) fn from_backend(data: T) -> Self {
        let shape = data.shape().to_vec();
        Tensor {
            data,
            mother_op: None,
            parameter_id: None,
            grad: Grad::none(),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            data: T::zeros(shape),
            mother_op: None,
            parameter_id: None,
            grad: Grad::none(),
        }
    }

    pub fn rand(shape: &[usize]) -> Self {
        Tensor {
            data: T::rand(shape),
            mother_op: None,
            parameter_id: None,
            grad: Grad::none(),
        }
    }

    pub fn new_trainable(val: &[f32], id: String) -> Self {
        Tensor {
            data: T::from_slice(val),
            mother_op: None,
            parameter_id: Some(id),
            grad: Grad::none(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub(in crate) fn set_grad(&self, grad: Tensor<T>) {
        assert_eq!(
            self.shape(), grad.shape(),
            "Gradient must have same shape as the Tensor itself: {:?} {:?}",
            self.shape(), grad.shape()
        );

        self.grad.set(grad);
    }

    /// Removes the computation graph history which generated this tensor
    pub fn into_standalone(mut self) -> Tensor<T> {
        self.mother_op = None;
        self
    }

//    pub fn reshape(self, shape: &[usize]) -> Tensor<T> {
//        crate::ops_impls::reshape(self, shape)
//    }

    pub fn enable_training(&mut self, id: String) {
        self.parameter_id = Some(id);
    }

    pub fn disable_training(&mut self) {
        self.parameter_id = None;
    }

    pub fn clone_only_data(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            mother_op: None,
            parameter_id: None,
            grad: Grad::none(),
        }
    }

    pub fn clone_without_op_graph(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            mother_op: None,
            parameter_id: self.parameter_id.clone(),
//            grad: self.grad.as_ref().map(|e| Box::new(e.clone_only_data())),
            grad: self.grad.clone(),
        }
    }

    pub fn backwards(&mut self, initial_grad: Option<Tensor<T>>) -> ParameterStore<T> {
        if let Some(grad) = initial_grad {
            assert_eq!(
                self.shape(), grad.shape(),
                "Gradient needs to have the same shape as the tensor itself"
            );
            self.grad.set(grad);// = Some(Box::new(grad));
        } else {
            assert!(
                self.shape().len() == 1 && self.shape()[0] == 1,
                "Backward call needs a gradient if the Tensor shape is not [1]"
            );
            self.set_grad(Tensor::new(&[1.]));
        }

        // where the parameters will be stored after having the gradients populated
        let mut hash: HashMap<String, Tensor<T>> = HashMap::new();

        // for each operation, calculate operands gradients recursively
        self.recursive_calc_grads(&mut hash);

        ParameterStore::from_hashmap(hash)
    }

    pub fn recursive_calc_grads(&self, hash: &mut HashMap<String, Tensor<T>>) {
        if let Some(op) = &self.mother_op {
            // Set the gradient of this tensor's original Op arguments
            let self_grad = self
                .grad.clone().0.take()
                .expect("In recursive_calc_grads without grad");

            op.set_operand_grad(&self_grad);

            // Ask the operands to set their Ops operands gradients too
            for operand in op.operands() {
                operand.recursive_calc_grads(hash);
            }

            // Now all gradients should have been updated
            let operands = op.operands();

            // Get all tensors which are parameters
            let trainable_operands: Vec<Tensor<T>> = operands
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


/*
    Broader problem: we need a way to index a Tensor, perform an operation on the indexed result,
    and make it reflect on the gradients of the original Tensor
    Something like
    x = [1, 2, 3, 4];
    y = 2*x[0]; // if x[0] borrows x
    z = 2*x[1]; // this here will not work

    (z+y).backward()

    Only one index can be active at a given time, so indexing needs to either move or borrow mutably
    but borrowing mutably it wouldn't be able to performs operations on the indexed values, so we
    need to move.

    Maybe something like PyTorch's narrow, which is a Tensor view into the original
*/


use std::ops::Index;
use std::cell::Cell;
use std::fmt::{Debug, Formatter, Error};

impl<T: TensorBackend> Index<&[usize]> for Tensor<T> {
    type Output = Tensor<T>;

    fn index(&self, index: &[usize]) -> &Self::Output {
        unimplemented!()
    }
}

