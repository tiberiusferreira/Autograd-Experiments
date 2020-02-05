use crate::ndarray_specific::mm_ndarray;
use crate::{Op, Tensor, TensorBackend};
use ndarray::prelude::IxDyn;

#[derive(Debug)]
pub struct MatMulOp<T: TensorBackend> {
    left: Tensor<T>,
    right: Tensor<T>,
}

pub fn matmul<T: TensorBackend>(left: Tensor<T>, right: Tensor<T>) -> Tensor<T> {
    let mul_op: MatMulOp<T> = MatMulOp { left, right };
    mul_op.forward()
}

impl<T: TensorBackend> Op<T> for MatMulOp<T> {
    fn name(&self) -> String {
        "MatMul".to_string()
    }

    fn forward(self) -> Tensor<T> {
        let shape_left =  &self.left.shape;
        let shape_right =  &self.right.shape;
        assert!(
            shape_left.len() == 2 && shape_right.len() == 2,
            "Can only multiply tensors of rank 2, but got {:?} and {:?}",
            shape_left,
            shape_right
        );
        assert_eq!(
            shape_left.last().unwrap(),
            shape_right.first().unwrap(),
            "Tensor shapes dont match for multiplication: {:?} and {:?}",
            shape_left,
            shape_right
        );
        let result = self.left.data.matmul2d(&self.right.data);
        let out_shape = &[shape_left[0], shape_right[1]];
        Tensor::from_op_result(result, out_shape, Box::new(self))
    }

    fn set_operand_grad(&mut self, previous_op_grad: Tensor<T>) {
        // taken from https://github.com/pytorch/pytorch/blob/master/tools/autograd/templates/Functions.cpp
        // left = mm_mat1_backward
        // right = mm_mat2_backward

//        self.left.grad = Some(Box::new(Tensor::from_ndarray(mm_ndarray(
//            previous_op_grad.data.view(),
//            self.right.data.t(),
//        ))));
//        self.right.grad = Some(Box::new(Tensor::from_ndarray(mm_ndarray(
//            self.left.data.t(),
//            previous_op_grad.data.view(),
//        ))));
        unimplemented!()
    }

    fn operands(&self) -> Vec<&Tensor<T>> {
        vec![&self.left, &self.right]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.left, &mut self.right]
    }

}
