// use crate::{GradFn, OpData, Tensor};
// use crate::tensor_backends::TensorBackend;
//
// pub fn index<'t, T: TensorBackend>(tensor: &Tensor<'t, T>, index: &[usize]) -> Tensor<'t, T> {
//     let grad_fn: GradFn<T> = GradFn(Box::new(
//         move |child_grad: T, self_grad: &mut T| {
//             self_grad.index(index) += child_grad[[0]];
//         },
//     ));
//
//     let operand_blueprint = tensor.self_gradient_blueprint(grad_fn);
//
//     let blueprints = vec![operand_blueprint];
//
//     let op_data = OpData::from_blueprints(blueprints, "Index".to_string());
//     Tensor {
//         tape: tensor.tape,
//         value: tensor.value.index(index),
//         parent_op_index: tensor.tape.push_op(op_data),
//     }
// }