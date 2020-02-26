use std::fmt::Debug;
use ndarray::Array;

mod ndarray_backend;
use ndarray::prelude::IxDyn;

pub mod indexing;

pub trait TensorBackend: Sized + Clone + Debug + 'static{
    /* Constructors, there are proxies to these in the Tape */
    fn from_slice(slice: &[f32]) -> Self;
    fn zeros(shape: &[usize]) -> Self;
    fn rand(shape: &[usize]) -> Self;
    fn zeros_like(other: &Self) -> Self;
    fn empty() -> Self;

    /* Helper functions */
    fn is_empty(&self) -> bool;
    fn fill_with(&mut self, value: f32);

    /* Shape Changing functions */
    /// Transposes dim 0 and 1, panics if they don't exist
    fn t(&mut self);
    fn reshape(&mut self, shape: &[usize]);
    fn shape(&self) -> &[usize];



    /* Basic Ops */
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;

    /* Basic Ops Scalar */
    fn add_scalar(&self, rhs: f32) -> Self;
    fn sub_scalar(&self, rhs: f32) -> Self;
    fn mul_scalar(&self, rhs: f32) -> Self;

    /// sums all elements
    fn sum(&self) -> f32;
    fn matmul2d(&self, rhs: &Self) -> Self;
    fn map_inplace<F>(&mut self, f: F) where F: FnMut(&mut f32);

    fn new_from_index(&self, index: &[usize]) -> Self;
    fn index(&self, index: &[usize]) -> f32;
    fn _index_mut(&mut self, index: &[usize]) -> &mut f32;
}

#[derive(Debug, Clone, PartialEq)]
pub struct NdArray(Array<f32, IxDyn>);