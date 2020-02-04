use crate::{Op, Tensor};
use ndarray::prelude::Dim;
use ndarray::{ArrayBase, IxDynImpl, OwnedRepr};

#[derive(Debug)]
pub struct AddScalar {
    left: Tensor,
    right: f32,
}

pub fn add_scalar(left: Tensor, right: f32) -> Tensor {
    AddScalar { left, right }.forward()
}

impl Op for AddScalar {
    fn name(&self) -> String {
        "AddScalar".to_string()
    }

    fn forward(self) -> Tensor {
        let out_data = &self.left.data + self.right;
        let mut output = Tensor::from_ndarray(out_data);
        output.mother_op = Some(Box::new(self));
        output
    }

    fn set_operand_grad(&mut self, previous_op_grad: Tensor) {
        self.left.set_grad(previous_op_grad.clone_only_data());
    }

    fn operands(&self) -> Vec<&Tensor> {
        vec![&self.left]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.left]
    }

}

impl std::ops::Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        let left = self;
        let right = rhs;
        let add_op: AddScalar = AddScalar { left, right };
        add_op.forward()
    }
}
