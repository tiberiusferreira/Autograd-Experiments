use ndarray::prelude::*;
use crate::ndarray_specific::IxDynActualType::OneD;
use crate::ops_impls::ndarray_specific::IxDynActualType::TwoD;

pub fn mm_ndarray(m1: &ndarray::Array<f32, IxDyn>, m2: &ndarray::Array<f32, IxDyn>) -> ndarray::Array<f32, IxDyn>{

    let m1 = get_actual_type(m1);
    let m2 = get_actual_type(m2);
    use IxDynActualType;
    match (&m1, &m2){
        (OneD(x), OneD(y)) => {
            ndarray::arr1(&[x.dot(y)]).into_dyn()
        },
        (OneD(x), TwoD(y)) => {
            x.dot(y).into_dyn()
        },
        (TwoD(x), OneD(y)) => {
            x.dot(y).into_dyn()
        },
        (TwoD(x), TwoD(y)) => {
            x.dot(y).into_dyn()
        }
    }

}

enum IxDynActualType<'a>{
    OneD(ndarray::ArrayView<'a, f32, Ix1>),
    TwoD(ndarray::ArrayView<'a, f32, Ix2>)
}

fn get_actual_type(m1: &ndarray::Array<f32, IxDyn>) -> IxDynActualType {
    match m1.shape().len(){
        1 => {
            let m1: ndarray::ArrayView<f32, Ix1> = m1.view().into_dimensionality().unwrap();
            IxDynActualType::OneD(m1)
        },
        2 => {
            let m1: ndarray::ArrayView<f32, Ix2> = m1.view().into_dimensionality().unwrap();
            IxDynActualType::TwoD(m1)
        }
        x => {
            panic!("Tried to multiply Tensor of rank: {}", x);
        }
    }
}