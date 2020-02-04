use crate::{Op, Tensor};
use ndarray::prelude::Dim;
use ndarray::{ArrayBase, IxDynImpl, OwnedRepr};

#[derive(Debug)]
pub struct Add {
    left: Tensor,
    right: Tensor,
}

pub fn add(left: Tensor, right: Tensor) -> Tensor {
    Add { left, right }.forward()
}

impl Op for Add {
    fn name(&self) -> String {
        "Add".to_string()
    }

    fn forward(self) -> Tensor {
        let out_data = &self.right.data + &self.left.data;
        let mut output = Tensor::from_ndarray(out_data);
        output.mother_op = Some(Box::new(self));
        output
    }

    fn set_operand_grad(&mut self, previous_op_grad: Tensor) {
        self.left.set_grad(previous_op_grad.clone_only_data());
        self.right.set_grad(previous_op_grad.clone_only_data());
    }

    fn operands(&self) -> Vec<&Tensor> {
        vec![&self.left, &self.right]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.left, &mut self.right]
    }

}

impl std::ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let left = self;
        let right = rhs;
        let add_op: Add = Add { left, right };
        add_op.forward()
    }
}
