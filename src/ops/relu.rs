use crate::{GradFn, OpData, TrackedTensor};
use crate::tensor_backends::TensorBackend;


pub fn relu<'t, T: TensorBackend>(input: &TrackedTensor<'t, T>) -> TrackedTensor<'t, T> {
    let mut closure_input_data_clone = input.data().clone();
    let grad_fn: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            // If value < 0 => grad = 0.1
            // else grad = 1.
            let mut input_data = closure_input_data_clone.clone();
            input_data.map_inplace(|single_data|{
                if *single_data < 0.{
                    *single_data = 0.1;
                }else{
                    *single_data = 1.;
                }
            });
            let grad = child_grad.mul(&input_data);
            *self_grad = self_grad.add(&grad);
        },
    ));

    let blueprint = input.self_gradient_blueprint(grad_fn);

    let op_data =
        OpData::from_blueprints(vec![blueprint], "Relu".to_string());

    let mut input_data_clone = input.data().clone();
    input_data_clone.map_inplace(|single_data|{
        if *single_data < 0.{
            *single_data = *single_data*0.1;
        }
    });

    let op_result = input_data_clone;
    input.tape.tensor_from_op_result_and_data(op_result, op_data)
}


#[cfg(test)]
mod relu_tests {
    use super::*;
    use crate::ops::testing::validate_grad;
    use crate::ops::*;
    use crate::tensor_backends::NdArray;
    use crate::tape::ComputationRecord;


    fn relu_comp<'t>(input: &TrackedTensor<'t, NdArray>) -> TrackedTensor<'t, NdArray> {
        let y = relu(&input);
        sum(&y)
    }

    #[test]
    fn relu_test() {
        let t: ComputationRecord<NdArray> = ComputationRecord::new();
        let input_0 = t.tensor_from_slice(&[-2., -1., 0., 1., 2.]);
        validate_grad(input_0, &relu_comp);
    }
}
