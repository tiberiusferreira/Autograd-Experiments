use ndarray::{arr1, ArrayBase};
use crate::{TensorBackend, NdArray};
mod matmul2d;

impl TensorBackend for NdArray {
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

    fn matmul2d(&self, rhs: &Self) -> Self {
        let self_view = self.0.view();
        let other_view = rhs.0.view();
        Self(matmul2d::mm_ndarray(self_view, other_view))
    }

    fn t(&self) -> Self {
        let mut new = self.clone();
        new.0.swap_axes(0, 1);
        new
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn sum(&self) -> f32 {
        self.0.sum()
    }

    fn fill_with(&mut self, value: f32) {
        self.0.fill(value);
    }
}
