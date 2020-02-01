use crate::{Op, Tensor};
use crate::ndarray_specific::mm_ndarray;
use ndarray::prelude::IxDyn;

#[derive(Debug)]
pub struct MatMulOp {
    left: Tensor,
    right: Tensor,
}

pub fn matmul(left: Tensor, right: Tensor) -> Tensor {
    let mul_op: MatMulOp = MatMulOp { left, right };
    mul_op.forward()
}

impl Op for MatMulOp {
    fn name(&self) -> String {
        "MatMul".to_string()
    }

    fn forward(self) -> Tensor {
        let data = mm_ndarray(self.left.data.view(), self.right.data.view());
        let mut tensor = Tensor::from_ndarray(data);
        tensor.mother_op = Some(Box::new(self));
        tensor
    }

    fn set_operand_grad(&mut self, previous_op_grad: ndarray::Array<f32, IxDyn>) {
        // taken from https://github.com/pytorch/pytorch/blob/master/tools/autograd/templates/Functions.cpp
        // left = mm_mat1_backward
        // right = mm_mat2_backward

        self.left.grad = Some(mm_ndarray(previous_op_grad.view(), self.right.data.t()));
        self.right.grad = Some(mm_ndarray(self.left.data.t(), previous_op_grad.view()));

    }

    fn operands(&self) -> Vec<&Tensor> {
        vec![&self.left, &self.right]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.left, &mut self.right]
    }

    fn operands_shallow_clone(&self) -> Vec<Tensor> {
        vec![self.left.shallow_clone(), self.right.shallow_clone()]
    }
}

//impl std::ops::Mul for Tensor {
//    type Output = Tensor;
//
//    fn mul(self, rhs: Self) -> Self::Output {
//        let left = self;
//        let right = rhs;
//        let mul_op: MatMulOp = MatMulOp { left, right };
//        mul_op.forward()
//    }
//}
