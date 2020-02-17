use crate::{Op, Tensor, TensorBackend};
use ndarray::prelude::Dim;
use ndarray::{ArrayBase, IxDynImpl, OwnedRepr};
use std::ops::Add;

#[derive(Debug)]
pub struct Sum<T: TensorBackend> {
    input: Tensor<T>,
}

pub fn sum<T: TensorBackend>(input: Tensor<T>) -> Tensor<T> {
    Sum { input }.forward()
}

impl<T: TensorBackend> Op<T> for Sum<T> {
    fn name(&self) -> String {
        "Sum".to_string()
    }

    fn forward(self) -> Tensor<T> {
        let sum = self.input.data.sum();
        let mut output = Tensor::new(&[sum]);
        output
    }

    fn set_operand_grad(&self, previous_op_grad: &Tensor<T>) {
        let operand_size = self.input.shape();
        let mut t = Tensor::<T>::zeros(operand_size);
        assert!(previous_op_grad.shape() == [1]);
        // Here previous_op_grad should be a scalar Tensor and we need to add that to t
        // the gradient is just a tensor, with shape equal to self and value equal to previous_op_grad

        // to make it work, we need indexing working
//        t.data.fill_with(previous_op_grad.data[0]);

        // TODO: Problem adding a "scalar" tensor should always be possible, but should the Tensor
        // be "cast" to a scalar?
//        t.add(previous_op_grad.)


        unimplemented!()
    }

    fn operands(&self) -> Vec<&Tensor<T>> {
        unimplemented!()
    }

//    fn operands_mut(&mut self) -> Vec<&mut Tensor<T>> {
//        unimplemented!()
//    }

}
