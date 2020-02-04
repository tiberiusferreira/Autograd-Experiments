mod ops_impls;
mod store;
pub use ops_impls::*;
use std::collections::HashMap;
pub use store::ParameterStore;
pub mod test_helpers;
use ndarray::prelude::arr1;
use ndarray::IxDyn;
mod tensor;
pub use tensor::{TensorBackend, NdArray};
pub use crate::tensor::Tensor;

//
//#[test]
//pub fn test(){
//    let a = ndarray::arr1(&[2, 1]);
//    let b = ndarray::arr1(&[2, 1]);
////    let a = a.into_shape((1, 2)).unwrap();
////    a.do
////    let a = a.into_dyn();
////    let b = b.into_dyn();
//    use ndarray::Ix1;
//    use ndarray::Ix2;
//    let w = a.into_dimensionality::<Ix2>().unwrap();
////    println!("{}", w);
////    let k = a * b;
////    use ndarray::Ix1;
////    let d: Array1<OwnedRepr<f32>> = c.try_into().unwrap();
//}
