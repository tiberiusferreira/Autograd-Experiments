use crate::{GradFn, OpData, TrackedTensor};
use crate::tensor_backends::TensorBackend;

//noinspection DuplicatedCode
pub fn sum<'t, T: TensorBackend>(input: &TrackedTensor<'t, T>) -> TrackedTensor<'t, T> {

    let grad_fn: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            // child_grad will be a scalar (since the output is a scalar)
            // here we create a tensor of the same shape as self_grad
            // fill it with child_grad and sum it
            let mut new = T::zeros_like(self_grad);
            new.fill_with(child_grad.index(&[0]));
            *self_grad = self_grad.add(&new);
        },
    ));
    let grad_blueprint = input.self_gradient_blueprint(grad_fn);

    let op_data =
        OpData::from_blueprints(vec![grad_blueprint], "Sum".to_string());

    let op_result = T::from_slice(&[input.data().sum()]);

    input.tape.tensor_from_op_result_and_data(op_result, op_data)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::*;
    use crate::tensor_backends::NdArray;
    use crate::ops::testing::validate_grad;

    fn sum_computation<'t>(input: &TrackedTensor<'t, NdArray>) -> TrackedTensor<'t, NdArray> {
        let y = sum(&input);
        y
    }

    #[test]
    fn sum_test() {
        let t: ComputationRecord<NdArray> = ComputationRecord::new();
        let input_0 = t.tensor_from_slice(&[1., 2., 3., 4.]);
        validate_grad(input_0, &sum_computation);
    }
}
