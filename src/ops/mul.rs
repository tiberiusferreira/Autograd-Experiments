use crate::{GradFn, OpData, Tensor};
use crate::tensor_backends::TensorBackend;

//noinspection DuplicatedCode
pub fn mul<'t, T: TensorBackend>(left: &Tensor<'t, T>, other: &Tensor<'t, T>) -> Tensor<'t, T> {
    let right_val = other.data().clone();
    let grad_fn_left: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            *self_grad = self_grad.add(&right_val.mul(&child_grad));
        },
    ));

    let left_val = left.data().clone();
    let grad_fn_right: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            *self_grad = self_grad.add(&left_val.mul(&child_grad));
        },
    ));

    let left_blueprint = left.self_gradient_blueprint(grad_fn_left);
    let right_blueprint = other.self_gradient_blueprint(grad_fn_right);

    let op_data =
        OpData::from_blueprints(vec![left_blueprint, right_blueprint], "Mul".to_string());

    let op_result = left.data().mul(&other.data());

    left.tape.new_from_op_result_and_data(op_result, op_data)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::*;
    use crate::tensor_backends::NdArray;
    use crate::ops::testing::validate_grad;
    use crate::ops::sum::sum;

    fn mul_twice<'t>(input: &Tensor<'t, NdArray>) -> Tensor<'t, NdArray> {
        // [2.] + in + in
        let input_1 = input.tape.new_tensor_from_slice(&[2.]);
        let x = mul(&input, &input_1);
        let y = mul(&x, &input);
        y
    }

    fn mul_twice_sum<'t>(input: &Tensor<'t, NdArray>) -> Tensor<'t, NdArray> {
        // [2.] + in + in
        let mut input_1 = NdArray::zeros_like(input.data());
        input_1.fill_with(2.);
        let input_1 = input.tape.new_from_backend_value(input_1);


        let x = mul(&input, &input_1);
        let y = mul(&x, &input);
        let y = sum(&y);
        y
    }

    #[test]
    fn mul_test() {
        let t: Tape<NdArray> = Tape::new();
        let scalar_input = t.new_tensor_from_slice(&[1.]);
        validate_grad(scalar_input, &mul_twice);

        let input = t.new_tensor_from_slice(&[1., 2., 3., 4.]);
        validate_grad(input, &mul_twice_sum);
    }
}
