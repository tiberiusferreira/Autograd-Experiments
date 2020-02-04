use super::*;
use crate::test_helpers::assert_very_close_in_value;
use crate::Tensor;

/// The idea here is to test the OPs gradients calculation.
/// To do so we can calculate the gradient with respect to the input
/// change the input a little bit and check that the output changes by gradient*(little bit)

fn test_binary_op(op: Box<dyn Fn(Tensor, Tensor) -> Tensor>) {
    let a_val = &[2., 3., 4., 5.];
    let grad_val = ndarray::arr2(&[[1., 1.], [1., 1.]]).into_dyn();
    let a = Tensor::new_trainable(a_val, "a".to_string());

    let mut b = reshape(a, &[2, 2]);
    b.backwards(Some(Tensor::from_ndarray(grad_val)));

    println!("{:#?}", b);
    //    let b = Tensor::new(b_val);
    //    let mut c = op(a, b);
    //    let mut vs = c.backwards();
    //
    //    let grad_a = vs.get("a").unwrap().grad.unwrap();
    //    let old_c = c;
    //
    //    let a = Tensor::new_trainable(a_val + delta, "a".to_string());
    //    let b = Tensor::new(b_val);
    //    let c = op(a, b);
    //    assert_very_close_in_value(c.data, old_c.data + delta*grad_a);
    //
    //
    //    let a_val = left_val;
    //    let b_val = right_val;
    //    let delta = 1e-3;
    //
    //
    //    let a = Tensor::new_trainable(a_val, "a".to_string());
    //    let b = Tensor::new(b_val);
    //    let mut c = op(b, a);
    //    let mut vs = c.backwards();
    //
    //    let grad_a = vs.get("a").unwrap().grad.unwrap();
    //    let old_c = c;
    //
    //    let a = Tensor::new_trainable(a_val + delta, "a".to_string());
    //    let b = Tensor::new(b_val);
    //    let c = op(b, a);
    //    assert_very_close_in_value(c.data, old_c.data + delta*grad_a);
}

//#[test]
//fn test_add() {
//    test_binary_op(2., 3., Box::new(add));
//}

//#[test]
//fn test_sub() {
//    test_binary_op(2., 3., Box::new(sub));
//}

#[test]
fn test_matmul() {
    test_binary_op(Box::new(matmul));
}

//#[test]
//fn test_relu() {
//    let a = Tensor::new_trainable(2., "a".to_string());
//    let mut c = relu(a);
//    let mut vs = c.backwards();
//
//    let grad_a = vs.get("a").unwrap().grad.unwrap();
//    let old_c = c;
//
//    let delta = 1e-3;
//    let a = Tensor::new_trainable(2. + delta, "a".to_string());
//    let c = relu(a);
//    assert_very_close_in_value(c.data, old_c.data + delta*grad_a);
//}
//
