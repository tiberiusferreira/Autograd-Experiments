use ndarray::prelude::*;


pub trait TensorBackend: Sized + Clone{
    fn from_slice(slice: &[f32]) -> Self;
    fn zeros(shape: &[usize]) -> Self;
    fn rand(shape: &[usize]) -> Self;
}

#[derive(Clone)]
pub struct NdArray(pub ndarray::Array<f32, IxDyn>);


impl TensorBackend for NdArray{
    fn from_slice(slice: &[f32]) -> Self {
        Self(arr1(slice).into_dyn())
    }

    fn zeros(shape: &[usize]) -> Self {
        Self(ArrayBase::zeros(shape).into_dyn())
    }

    fn rand(shape: &[usize]) -> Self {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let dist = Uniform::new(0., 10.);
        Self(ndarray::Array::random(shape, dist).into_dyn())
    }
}