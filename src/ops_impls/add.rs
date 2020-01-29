use crate::{Tensor, Op};

#[derive(Debug)]
pub struct Add {
    left: Tensor,
    right: Tensor,
}

pub fn add(left: Tensor, right: Tensor) -> Tensor{
    Add{
        left,
        right
    }.forward()
}

impl Op for Add {
    fn name(&self) -> String {
        "Add".to_string()
    }

    fn forward(self) -> Tensor {
        let data = self.left.data + self.right.data;
        let mut tensor = Tensor::new(data);
        tensor.mother_op = Some(Box::new(self));
        tensor
    }


    fn set_operand_grad(&mut self, previous_op_grad: f32) {
        self.right.grad = Some(previous_op_grad);
        self.left.grad = Some(previous_op_grad);
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

impl std::ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let left = self;
        let right = rhs;
        let add_op: Add = Add {
            left,
            right
        };
        add_op.forward()
    }
}

