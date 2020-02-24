use crate::{GradFn, OpData, Tensor};
use crate::tensor_backends::TensorBackend;

//noinspection DuplicatedCode
pub fn add<'t, T: TensorBackend>(left: &Tensor<'t, T>, other: &Tensor<'t, T>) -> Tensor<'t, T> {

    let left_grad_fn_add: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            *self_grad = self_grad.add(&child_grad);
        },
    ));

    let right_grad_fn_add: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            *self_grad = self_grad.add(&child_grad);
        },
    ));


    let left_blueprint = left.self_gradient_blueprint(left_grad_fn_add);
    let right_blueprint = other.self_gradient_blueprint(right_grad_fn_add);

    let op_data =
        OpData::from_blueprints(vec![left_blueprint, right_blueprint], "Add".to_string());

    let op_result = left.data().add(&other.data());
    left.tape.new_from_op_result_and_data(op_result, op_data)
}


#[cfg(test)]
mod add_tests {
    use super::*;
    use crate::ops::testing::validate_grad;
    use crate::tensor_backends::NdArray;
    use crate::tape::Tape;


    fn add_twice<'t>(input: &Tensor<'t, NdArray>) -> Tensor<'t, NdArray> {
        // [2.] + in + in
        let input_1 = input.tape.new_tensor_from_slice(&[2.]);
        let x = add(&input, &input_1);
        let y = add(&x, &input);
        y
    }

    #[test]
    fn add_test() {
        let t: Tape<NdArray> = Tape::new();
        let input_0 = t.new_tensor_from_slice(&[1.]);
        validate_grad(input_0, &add_twice);
    }
}
