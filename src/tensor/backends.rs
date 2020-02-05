use std::fmt::Debug;
use ndarray::Array;

mod ndarray_backend;
use ndarray::prelude::IxDyn;

pub trait TensorBackend: Sized + Clone + Debug + 'static{
    fn from_slice(slice: &[f32]) -> Self;
    fn zeros(shape: &[usize]) -> Self;
    fn rand(shape: &[usize]) -> Self;
    fn matmul2d(&self, rhs: &Self) -> Self;
}

#[derive(Debug, Clone)]
pub struct NdArray(pub Array<f32, IxDyn>);
