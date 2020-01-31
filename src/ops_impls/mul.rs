use crate::{Op, Tensor};
use ndarray::Ix2;
#[derive(Debug)]
pub struct MulOp {
    left: Tensor,
    right: Tensor,
}

pub fn mul(left: Tensor, right: Tensor) -> Tensor {
    let mul_op: MulOp = MulOp { left, right };
    mul_op.forward()
}

impl Op for MulOp {
    fn name(&self) -> String {
        "Mul".to_string()
    }

    fn forward(mut self) -> Tensor {
        let left_shape = self.left.shape.clone();
        let right_shape = self.right.shape.clone();

        if left_shape.len() > 2 || right_shape.len() > 2 {
            panic!("Mul is only implemented for 1D or 2D tensors");
        }


//        if left_shape[1] != right_shape[0]  {
//            panic!("Left shape and right shape dont match for mul: Left: {:?}  Right: {:?}", left_shape, right_shape);
//        }
//
//        let left_data;
//
//        if left_shape.len() == 2 {
//            left_data = self
//                .left
//                .data
//                .into_dimensionality::<Ix2>()
//                .expect("Left mul argument was not 2D");
//        }else{
//            left_data = self
//                .left
//                .data
//                .into_dimensionality::<Ix1>()
//                .expect("Left mul argument was not 1D");
//        }
//        use ndarray::Ix1;
//        let right_data;
//
//        if right_shape.len() == 2{
//            right_data = self
//                .right
//                .data
//                .into_dimensionality::<Ix2>()
//                .expect("Right mul argument was not 2D");
//        }else{
//            right_data = self
//                .right
//                .data
//                .into_dimensionality::<Ix1>()
//                .expect("Right mul argument was not 1D");
//        }
//
//
//
//        let data = left_data.dot(&right_data).into_dyn();
//
//        self.left.data = left_data.into_dyn();
//        self.right.data = right_data.into_dyn();
//
//        let mut tensor = Tensor::from_ndarray(data);
//        tensor.mother_op = Some(Box::new(self));
//        tensor
        unimplemented!()
    }

    fn set_operand_grad(&mut self, previous_op_grad: f32) {
//        self.right.grad = Some(self.left.data * previous_op_grad);
//        self.left.grad = Some(self.right.data * previous_op_grad);
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

impl std::ops::Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let left = self;
        let right = rhs;
        let mul_op: MulOp = MulOp { left, right };
        mul_op.forward()
    }
}
