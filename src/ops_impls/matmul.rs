use crate::{Op, Tensor, TensorBackend};
use ndarray::prelude::IxDyn;

#[derive(Debug)]
pub struct MatMulOp<'a, T: TensorBackend + 'static> {
    left: &'a Tensor<'a, T>,
    right: &'a Tensor<'a, T>,
}

pub fn matmul<'a, T: TensorBackend + 'static>(left: &'a Tensor<'a, T>, right: &'a Tensor<'a, T>){// -> Tensor<'a, T> {
    // let mul_op: MatMulOp<T> = MatMulOp { left, right };
    // mul_op.forward()
}

impl<'a, T: TensorBackend + 'static> Op<'a, T> for MatMulOp<'a, T> {
    fn name(&self) -> String {
        "MatMul".to_string()
    }

    fn forward(self) -> Tensor<'a, T> {
        // let shape_left =  &self.left.shape();
        // let shape_right =  &self.right.shape();
        // assert!(
        //     shape_left.len() == 2 && shape_right.len() == 2,
        //     "Can only multiply tensors of rank 2, but got {:?} and {:?}",
        //     shape_left,
        //     shape_right
        // );
        // assert_eq!(
        //     shape_left.last().unwrap(),
        //     shape_right.first().unwrap(),
        //     "Tensor shapes dont match for multiplication: {:?} and {:?}",
        //     shape_left,
        //     shape_right
        // );
        // let result = self.left.data.matmul2d(&self.right.data);
        // Tensor::from_op_result(result, Box::new(self))
        unimplemented!()
    }

    fn set_operand_grad(&self, previous_op_grad: &Tensor<T>) {
        // taken from https://github.com/pytorch/pytorch/blob/master/tools/autograd/templates/Functions.cpp
        // left = mm_mat1_backward
        // right = mm_mat2_backward

        // Here we dont need to check the shapes because if they were valid for the forwards path,
        // they are valid for the backward
        let right_transposed = self.right.data.t();

        let left_grad = previous_op_grad.data.matmul2d(&right_transposed);
//        self.left.grad = Some(Box::new(Tensor::from_backend(left_grad)));
        self.left.grad.set(Tensor::from_backend(left_grad));

        let right_grad = self.left.data.t().matmul2d(&previous_op_grad.data);
        self.right.grad.set(Tensor::from_backend(right_grad));
//        self.right.grad = Some(Box::new(Tensor::from_backend(right_grad)));
    }

    // fn operands<'c>(&'c self) -> Vec<&'c Tensor<T>>{
    //     let  l = self.right;
    //     let  r = self.right;
    //     vec![l, r]
    // }

//    fn operands_mut(&mut self) -> Vec<&mut Tensor<T>> {
//        vec![&mut self.left, &mut self.right]
//    }

}
