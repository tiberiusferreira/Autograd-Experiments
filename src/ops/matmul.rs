use crate::{GradFn, OpData, TrackedTensor};
use crate::tensor_backends::TensorBackend;

//noinspection DuplicatedCode
pub fn matmul<'t, T: TensorBackend>(left: &TrackedTensor<'t, T>, right: &TrackedTensor<'t, T>) -> TrackedTensor<'t, T> {

    let right_data = right.data().clone();
    let left_grad_fn_add: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            // println!("Child grad: {:?}", child_grad);
            // println!("Right: {:?}", right_data);
            let mut right_data_clone = right_data.clone();
            right_data_clone.t();
            // println!("Right T : {:?}", right_data_clone);
            let new_self_grad = child_grad.matmul2d(&right_data_clone);
            *self_grad = self_grad.add(&new_self_grad);
        },
    ));

    let left_data = left.data().clone();
    let right_grad_fn_add: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            let mut left_data_clone = left_data.clone();
            left_data_clone.t();
            let new_self_grad = left_data_clone.matmul2d(&child_grad);
            *self_grad = self_grad.add(&new_self_grad);
        },
    ));


    let left_blueprint = left.self_gradient_blueprint(left_grad_fn_add);
    let right_blueprint = right.self_gradient_blueprint(right_grad_fn_add);

    let op_data =
        OpData::from_blueprints(vec![left_blueprint, right_blueprint], "Matmul".to_string());

    let op_result = left.data().matmul2d(&right.data());
    left.tape.tensor_from_op_result_and_data(op_result, op_data)
}


#[cfg(test)]
mod matmul_tests {
    use super::*;
    use crate::ops::testing::validate_grad;
    use crate::tensor_backends::NdArray;
    use crate::tape::ComputationRecord;
    use crate::ops::sum::sum;


    fn matmul_compute<'t>(input: &TrackedTensor<'t, NdArray>) -> TrackedTensor<'t, NdArray> {
        let mut data = NdArray::from_slice(&[5., 6., 7., 8.]);
        data.reshape(&[2, 2]);
        let input_1 = input.tape.tensor_from_value(data);
        let x = matmul(&input, &input_1);
        let y = matmul(&x, &input);
        let z = sum(&y);
        z
    }

    #[test]
    fn matmul_test() {
        let t: ComputationRecord<NdArray> = ComputationRecord::new();
        let mut data = NdArray::from_slice(&[1., 2., 3., 4.]);
        data.reshape(&[2, 2]);
        let input_0 = t.tensor_from_value(data);
        validate_grad(input_0, &matmul_compute);
    }
}
