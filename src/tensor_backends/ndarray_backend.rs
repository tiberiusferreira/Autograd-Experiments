use ndarray::{arr1, ArrayBase, arr0};
use crate::tensor_backends::{TensorBackend, NdArray};

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

    fn zeros_like(other: &Self) -> Self {
        let shape = other.shape();
        Self::zeros(shape)
    }

    fn empty() -> Self {
        Self::from_slice(&[])
    }

    fn is_empty(&self) -> bool {
        self.shape() == &[] || self.shape() == &[0]
    }


    fn fill_with(&mut self, value: f32) {
        self.0.fill(value);
    }

    fn t(&mut self) {
        self.0.swap_axes(0, 1);
    }

    fn reshape(&mut self, shape: &[usize]) {
        self.0 = self.0.clone().into_shape(shape).expect("Invalid shape");
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "Can only add elements of same shape");
        Self(&self.0 + &rhs.0)
    }

    fn sub(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "Can only sub elements of same shape");
        Self(&self.0 - &rhs.0)
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self(&self.0*&rhs.0)
    }

    fn add_scalar(&self, rhs: f32) -> Self {
        Self(&self.0 + rhs)
    }


    fn sub_scalar(&self, rhs: f32) -> Self {
        Self(&self.0 - rhs)
    }

    fn mul_scalar(&self, rhs: f32) -> Self {
        Self(&self.0 * rhs)
    }

    fn sum(&self) -> f32 {
        self.0.sum()
    }

    fn matmul2d(&self, rhs: &Self) -> Self {
        let self_view = self.0.view();
        let other_view = rhs.0.view();
        Self(matmul2d::mm_ndarray(self_view, other_view))
    }

    fn map_inplace<F>(&mut self, f: F) where F: FnMut(&mut f32) {
        self.0.map_inplace(f);
    }

    fn new_from_index(&self, index: &[usize]) -> Self {
        let indexes = index.to_vec();
        let inner = &self.0;
        let indexed_val = match indexes.len(){
            0 => {panic!("Invalid index: &[]")},
            1 => arr0(inner[[indexes[0]]]).into_dyn(),
            2 => arr0(inner[[indexes[0], indexes[1]]]).into_dyn(),
            3 => arr0(inner[[indexes[0], indexes[1], indexes[2]]]).into_dyn(),
            _ => panic!()
        };
        NdArray(indexed_val)
    }

    fn index(&self, index: &[usize]) -> f32 {
        assert_eq!(index.len(), self.shape().len(), "Needs to index with the same number of dimensions and the Tensor itself.");
        let indexes = index.to_vec();
        let inner = &self.0;
        let indexed_val = match indexes.len(){
            0 => {panic!("Invalid index: &[]")},
            1 => inner[[indexes[0]]],
            2 => inner[[indexes[0], indexes[1]]],
            3 => inner[[indexes[0], indexes[1], indexes[2]]],
            _ => panic!("Indexing not implemented for more than 3 dims")
        };
        indexed_val
    }

    fn _index_mut(&mut self, index: &[usize]) -> &mut f32 {
        let indexes = index.to_vec();
        let inner = &mut self.0;
        let indexed_val = match indexes.len(){
            0 => {panic!("Invalid index: &[]")},
            1 => &mut inner[[indexes[0]]],
            2 => &mut inner[[indexes[0], indexes[1]]],
            3 => &mut inner[[indexes[0], indexes[1], indexes[2]]],
            _ => panic!("Indexing not implemented for more than 3 dims")
        };
        indexed_val
    }
}


#[cfg(test)]
mod ndarray_backend_tests {
    use crate::tensor_backends::{NdArray, TensorBackend};
    use super::*;
    #[test]
    fn scalar_add() {
        let left = NdArray::from_slice(&[1., 2., 3.]);
        let right = 2.;
        assert_eq!(NdArray::from_slice(&[3., 4., 5.]), left.add_scalar(right));
        assert_eq!(left, NdArray::from_slice(&[1., 2., 3.]));
    }
}
