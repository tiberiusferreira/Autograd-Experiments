mod matmul;
use crate::tensor::Tensor;
pub use matmul::matmul;

mod reshape;
pub use reshape::reshape;

mod sum;

//mod relu;
//pub use relu::relu;
//pub use relu::relu_custom;
mod add;
pub use add::add;

mod add_scalar;
pub use add_scalar::add_scalar;
//mod sub;
//pub use sub::sub;
pub(in crate) mod ndarray_specific;
#[cfg(test)]
mod tests;
use ndarray::prelude::IxDyn;

pub trait Op: std::fmt::Debug {
    fn name(&self) -> String;
    /// Should return the result of the operation.
    /// The Tensor returned should have this Op as its "mother op"
    fn forward(self) -> Tensor;
    /// Should set its own operands gradients
    fn set_operand_grad(&mut self, previous_op_grad: Tensor);
    fn operands(&self) -> Vec<&Tensor>;
    fn operands_mut(&mut self) -> Vec<&mut Tensor>;
}
