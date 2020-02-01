use ndarray::prelude::*;
use crate::ndarray_specific::IxDynActualType::OneD;
use crate::ops_impls::ndarray_specific::IxDynActualType::TwoD;

pub fn mm_ndarray(m1: ndarray::ArrayView<f32, IxDyn>, m2: ndarray::ArrayView<f32, IxDyn>) -> ndarray::Array<f32, IxDyn>{
    let shape_1 = m1.shape();
    let shape_2 = m2.shape();
    assert_eq!(shape_1.len(), shape_2.len(), "Cannot multiply Tensors of different ranks: {:?} and {:?}", shape_1, shape_2);
    assert_eq!(shape_1.last().unwrap(), shape_2.first().unwrap(), "Tensor shapes dont match for multiplication: {:?} and {:?}", shape_1, shape_2);

    let m1_enum = get_actual_type(m1.view());
    let m2_enum = get_actual_type(m2.view());
    match (&m1_enum, &m2_enum){
        (OneD(x), OneD(y)) => {
            ndarray::arr1(&[x.dot(y)]).into_dyn()
        },
        (TwoD(x), TwoD(y)) => {
            x.dot(y).into_dyn()
        },
        _ => {
            panic!("Cannot multiply Tensors of different ranks: {:?} and {:?}", m1.shape(), m1.shape());
        }
    }

}

enum IxDynActualType<'a>{
    OneD(ndarray::ArrayView<'a, f32, Ix1>),
    TwoD(ndarray::ArrayView<'a, f32, Ix2>)
}

fn get_actual_type(m1: ndarray::ArrayView<f32, IxDyn>) -> IxDynActualType {
    match m1.shape().len(){
        1 => {
            let m1: ndarray::ArrayView<f32, Ix1> = m1.into_dimensionality().unwrap();
            IxDynActualType::OneD(m1)
        },
        2 => {
            let m1: ndarray::ArrayView<f32, Ix2> = m1.into_dimensionality().unwrap();
            IxDynActualType::TwoD(m1)
        }
        x => {
            panic!("Tried to multiply Tensor of rank: {}", x);
        }
    }
}