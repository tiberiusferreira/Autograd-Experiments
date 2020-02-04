use crate::ndarray_specific::mm_ndarray;
use crate::{Op, Tensor};
use ndarray::prelude::Dim;
use ndarray::prelude::IxDyn;
use ndarray::{ArrayBase, IxDynImpl, OwnedRepr};

#[derive(Debug)]
pub struct Reshape {
    input: Tensor,
    output_shape: Vec<usize>,
}

pub fn reshape(input: Tensor, shape: &[usize]) -> Tensor {
    let reshape = Reshape {
        input,
        output_shape: shape.to_vec(),
    };
    reshape.forward()
}

impl Op for Reshape {
    fn name(&self) -> String {
        "Reshape".to_string()
    }

    fn forward(self) -> Tensor {
        let mut out_tensor = self.input.clone_only_data();
        out_tensor.data = out_tensor
            .data
            .into_shape(self.output_shape.clone())
            .expect("Error reshaping");
        out_tensor.shape = self.output_shape.clone();
        out_tensor.mother_op = Some(Box::new(self));
        out_tensor
    }

    fn set_operand_grad(&mut self, previous_op_grad: Tensor) {
        // example: [6] -> [2x3]
        // can we just reshape the gradients back to the original shape?
        // Lets try
        let a = previous_op_grad
            .data
            .into_shape(self.input.shape())
            .expect("Error reshaping gradients in backwards pass");
        self.input.grad = Some(Box::new(Tensor::from_ndarray(a)));
    }

    fn operands(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.input]
    }

}
