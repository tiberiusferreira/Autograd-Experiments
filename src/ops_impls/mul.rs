use crate::{Tensor, Op};

#[derive(Debug)]
pub struct MulOp {
    left: Tensor,
    right: Tensor,
}

pub fn mul(left: Tensor, right: Tensor) -> Tensor {
    let mul_op: MulOp = MulOp {
        left,
        right
    };
    mul_op.forward()
}

impl Op for MulOp {
    fn name(&self) -> String {
        "Mul".to_string()
    }

    fn forward(self) -> Tensor {
        let data = self.left.data*self.right.data;
        let mut tensor = Tensor::new(data);
        tensor.mother_op = Some(Box::new(self));
        tensor
    }


    fn set_operand_grad(&mut self, previous_op_grad: f32) {
        self.right.grad = Some(self.left.data*previous_op_grad);
        self.left.grad = Some(self.right.data*previous_op_grad);
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
        let mul_op: MulOp = MulOp {
            left,
            right
        };
        mul_op.forward()
    }
}

