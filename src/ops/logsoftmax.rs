use crate::{GradFn, OpData, TrackedTensor};
use crate::tensor_backends::TensorBackend;

//noinspection DuplicatedCode
pub fn logsoftmax<'t, T: TensorBackend>(input: &TrackedTensor<'t, T>) -> TrackedTensor<'t, T> {

    let input_data_closure = input.data().clone();

    let grad_fn_add: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {

            let sum_grad_output = child_grad.sum();

            println!("grad output: {:?}", sum_grad_output);
            println!("child_grad: {:?}", child_grad);
            let mut input_data_exp = input_data_closure.clone();

            input_data_exp.map_inplace(|in_data|{
                *in_data = (*in_data).exp();
            });

            let out = child_grad.sub( &input_data_exp.mul_scalar( sum_grad_output ) );


            *self_grad = (self_grad.add(&out));

        },
    ));


    let grad_blueprint = input.self_gradient_blueprint(grad_fn_add);

    let op_data =
        OpData::from_blueprints(vec![grad_blueprint], "LogSoftmax".to_string());


    let mut input_data = input.data().clone();

    let mut input_data_clone = input_data.clone();

    input_data_clone.map_inplace(|in_data|{
        *in_data = (*in_data).exp();
    });
    let sum_ln = input_data_clone.sum().ln();

    input_data.map_inplace(|in_data|{
        *in_data = (*in_data) - sum_ln;
    });

    let op_result = input_data;

    input.tape.tensor_from_op_result_and_data(op_result, op_data)
}


#[cfg(test)]
mod logsoftmax_tests {
    use super::*;
    use crate::ops::testing::validate_grad;
    use crate::tensor_backends::NdArray;
    use crate::tape::ComputationRecord;
    use crate::ops::sum::sum;


    fn logsoftmax_compute<'t>(input: &TrackedTensor<'t, NdArray>) -> TrackedTensor<'t, NdArray> {

        let soft_max = logsoftmax(input);
        println!("SM = {:?}", soft_max.data());
        sum(&soft_max)
        // let mut data = NdArray::from_slice(&[5., 6., 7., 8.]);
        // data.reshape(&[2, 2]);
        // let input_1 = input.tape.tensor_from_value(data);
        // let x = matmul(&input, &input_1);
        // let y = matmul(&x, &input);
        // let z = sum(&y);
        // z
    }

    #[test]
    fn logsoftmax_test() {

        let t: ComputationRecord<NdArray> = ComputationRecord::new();
        let mut data = NdArray::from_slice(&[1., 2., 3., 4.]);

        let input_0 = t.tensor_from_value(data);
        validate_grad(input_0, &logsoftmax_compute);

    }
}
