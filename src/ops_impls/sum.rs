use crate::{Op, Tensor};
use ndarray::prelude::Dim;
use ndarray::{ArrayBase, IxDynImpl, OwnedRepr};
use std::ops::Add;

#[derive(Debug)]
pub struct Sum {
    input: Tensor,
}

pub fn sum(input: Tensor) -> Tensor {
    Sum { input }.forward()
}

impl Op for Sum {
    fn name(&self) -> String {
        "Sum".to_string()
    }

    fn forward(self) -> Tensor {
        let sum = self.input.data.sum();
        let mut output = Tensor::new(&[sum]);
        output.shape = vec![1];
        output
    }

    fn set_operand_grad(&mut self, previous_op_grad: Tensor) {
        let operand_size = self.input.shape.as_slice();
        let mut t = Tensor::zeros(operand_size);
        assert!(previous_op_grad.shape.as_slice() == [1]);
        // Here previous_op_grad should be a scalar Tensor and we need to add that to t
        // the gradient is just a tensor, with shape equal to self and value equal to previous_op_grad



        // TODO: Problem adding a "scalar" tensor should always be possible, but should the Tensor
        // be "cast" to a scalar?
//        t.add(previous_op_grad.)


        unimplemented!()
    }

    fn operands(&self) -> Vec<&Tensor> {
        unimplemented!()
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor> {
        unimplemented!()
    }

}
