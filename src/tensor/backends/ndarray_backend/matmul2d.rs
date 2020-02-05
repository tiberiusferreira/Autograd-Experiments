use ndarray::prelude::*;

pub fn mm_ndarray(
    m1: ndarray::ArrayView<f32, IxDyn>,
    m2: ndarray::ArrayView<f32, IxDyn>,
) -> ndarray::Array<f32, IxDyn> {
    let shape_1 = m1.shape();
    let shape_2 = m2.shape();
    assert!(
        shape_1.len() == 2 && shape_2.len() == 2,
        "Can only multiply tensors of rank 2, but got {:?} and {:?}",
        shape_1,
        shape_2
    );
    assert_eq!(
        shape_1.last().unwrap(),
        shape_2.first().unwrap(),
        "Tensor shapes dont match for multiplication: {:?} and {:?}",
        shape_1,
        shape_2
    );

    let m1: ndarray::ArrayView<f32, Ix2> = m1.view().into_dimensionality().unwrap();
    let m2: ndarray::ArrayView<f32, Ix2> = m2.view().into_dimensionality().unwrap();
    m1.dot(&m2).into_dyn()
}
