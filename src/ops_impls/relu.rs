use crate::{Tensor, Op};

#[derive(Debug)]
pub struct Relu {
    operand: Tensor,
    leak_const: f32
}

pub fn relu(t: Tensor) -> Tensor{
    Relu{
        operand: t,
        leak_const: 0.1
    }.forward()
}

pub fn relu_custom(t: Tensor, neg_slope: f32) -> Tensor{
    Relu{
        operand: t,
        leak_const: neg_slope
    }.forward()
}

impl Op for Relu {
    fn name(&self) -> String {
        "ReLu".to_string()
    }

    fn forward(self) -> Tensor {
        let mut out_data= self.operand.data;
        if self.operand.data < 0.{
            out_data = self.operand.data*self.leak_const;
        }
        let mut out_tensor = Tensor::new(out_data);
        out_tensor.mother_op = Some(Box::new(self));
        out_tensor
    }

    fn set_operand_grad(&mut self, previous_op_grad: f32) {
        // 0 when input < 0, equal to input if input > 0
        let out_grad;
        if self.operand.data < 0. {
            out_grad = previous_op_grad*0.1;
        }else{
            out_grad = previous_op_grad;
        }
        self.operand.grad = Some(out_grad);
    }

    fn operands(&self) -> Vec<&Tensor> {
        vec![&self.operand]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.operand]
    }

    fn operands_shallow_clone(&self) -> Vec<Tensor> {
        vec![self.operand.shallow_clone()]
    }
}


