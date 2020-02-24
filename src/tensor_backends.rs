use std::fmt::Debug;
use ndarray::Array;

mod ndarray_backend;
use ndarray::prelude::IxDyn;

pub mod indexing;

pub trait TensorBackend: Sized + Clone + Debug + 'static{
    // Constructors, there are proxies to these in the Tape
    fn from_slice(slice: &[f32]) -> Self;
    fn zeros(shape: &[usize]) -> Self;
    fn rand(shape: &[usize]) -> Self;
    fn zeros_like(other: &Self) -> Self;
    fn empty() -> Self;

    fn is_empty(&self) -> bool;

    /// Transposes dim 0 and 1, panics if they don't exist
    fn t(&self) -> Self;
    fn shape(&self) -> &[usize];

    /// sums all elements
    fn sum(&self) -> f32;
    fn fill_with(&mut self, value: f32);

    fn matmul2d(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn new_from_index(&self, index: &[usize]) -> Self;
    fn index(&self, index: &[usize]) -> f32;
    fn _index_mut(&mut self, index: &[usize]) -> &mut f32;
    fn add(&self, rhs: &Self) -> Self;
}

#[derive(Debug, Clone)]
pub struct NdArray(Array<f32, IxDyn>);