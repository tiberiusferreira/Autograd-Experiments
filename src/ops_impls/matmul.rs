use crate::ndarray_specific::mm_ndarray;
use crate::{Op, Tensor};
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

    fn set_operand_grad(&mut self, previous_op_grad: Tensor) {
        // taken from https://github.com/pytorch/pytorch/blob/master/tools/autograd/templates/Functions.cpp
        // left = mm_mat1_backward
        // right = mm_mat2_backward

        self.left.grad = Some(Box::new(Tensor::from_ndarray(mm_ndarray(
            previous_op_grad.data.view(),
            self.right.data.t(),
        ))));
        self.right.grad = Some(Box::new(Tensor::from_ndarray(mm_ndarray(
            self.left.data.t(),
            previous_op_grad.data.view(),
        ))));
    }

    fn operands(&self) -> Vec<&Tensor<T>> {
        vec![&self.left, &self.right]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.left, &mut self.right]
    }

}
