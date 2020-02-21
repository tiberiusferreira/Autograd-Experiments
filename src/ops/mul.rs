use crate::{GradFn, OpData, Tensor};
use crate::tensor_backends::TensorBackend;

//noinspection DuplicatedCode
pub fn mul<'t, T: TensorBackend>(left: &Tensor<'t, T>, other: &Tensor<'t, T>) -> Tensor<'t, T> {
    let right_val = other.value.clone();
    let grad_fn_left: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            *self_grad = right_val.mul(&child_grad);
        },
    ));

    let left_val = left.value.clone();
    let grad_fn_right: GradFn<T> = GradFn(Box::new(
        move |child_grad: T, self_grad: &mut T| {
            *self_grad = left_val.mul(&child_grad);
        },
    ));

    let left_blueprint = left.self_gradient_blueprint(grad_fn_left);
    let right_blueprint = other.self_gradient_blueprint(grad_fn_right);

    let op_data =
        OpData::from_blueprints(vec![left_blueprint, right_blueprint], "Mul".to_string());

    Tensor {
        tape: left.tape,
        value: left.value.mul(&other.value()),
        parent_op_index: left.tape.push_op(op_data),
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::*;
    use crate::tensor_backends::NdArray;

    #[test]
    fn mul_test() {
            let t: Tape<NdArray> = Tape::new();
            let x = t.new_tensor(&[1., 2.]);
            let y = t.new_tensor(&[3., 4.]);
            let z = mul(&x, &y);
            let grad = z.grad();
            println!("{:#?}", grad);
            println!("{:#?}", grad.wrt(&x));
    }
}
