use crate::{Op, Tensor};
use crate::ndarray_specific::mm_ndarray;

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

    fn forward(self) -> Tensor {
//        if left_shape.len() > 2 || right_shape.len() > 2 {
//            panic!("Mul is only implemented for 1D or 2D tensors");
//        }
        let data = mm_ndarray(&self.left.data, &self.right.data);
        let mut tensor = Tensor::from_ndarray(data);
        tensor.mother_op = Some(Box::new(self));
        tensor

    }

    fn set_operand_grad(&mut self, _previous_op_grad: f32) {
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
