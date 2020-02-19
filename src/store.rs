//use crate::Tensor;
//use std::collections::HashMap;
//use crate::tensor::TensorBackend;
//
//#[derive(Debug)]
//pub struct ParameterStore<'a, T: TensorBackend> {
//    parameters: HashMap<String, Tensor<'a, T>>,
//}
//
//impl<'a, T: TensorBackend> ParameterStore<'a, T> {
//    pub fn insert(&mut self, id: String, t: Tensor<'a, T>) {
//        self.parameters.insert(id, t);
//    }
//    pub fn get(&mut self, id: &str) -> Option<&Tensor<'a, T>> {
//        self.parameters.get(id)
//    }
//    pub fn get_mut(&mut self, id: &str) -> Option<&mut Tensor<'a, T>> {
//        self.parameters.get_mut(id)
//    }
//    pub fn remove(&mut self, id: &str) -> Option<Tensor<'a, T>> {
//        self.parameters.remove(id)
//    }
//    pub fn remove_or_init(&mut self, id: &str, init_val: &[f32]) -> Tensor<'a, T>{
//        self.parameters
//            .remove(id)
//            .unwrap_or(Tensor::new_trainable(init_val, id.to_string()))
//    }
//    pub fn new() -> Self {
//        ParameterStore {
//            parameters: Default::default(),
//        }
//    }
//    pub fn from_hashmap(hash: HashMap<String, Tensor<'a, T>>) -> Self {
//        ParameterStore { parameters: hash }
//    }
//    pub fn into_raw(self) -> HashMap<String, Tensor<'a, T>> {
//        self.parameters
//    }
//}
//
//impl<'a, T: TensorBackend> IntoIterator for ParameterStore<'_, T> {
//    type Item = (String, Tensor<'a, T>);
//
//    type IntoIter = std::collections::hash_map::IntoIter<String, Tensor<'a, T>>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        self.parameters.into_iter()
//    }
//}
//
//impl<'a, T: TensorBackend> IntoIterator for &'a ParameterStore<'_, T> {
//    type Item = (&'a String, &'a Tensor<'a, T>);
//
//    type IntoIter = std::collections::hash_map::Iter<'a, String, Tensor<'a, T>>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        (&self.parameters).into_iter()
//    }
//}
//
//impl<'a, T: TensorBackend> IntoIterator for &'a mut ParameterStore<'_, T> {
//    type Item = (&'a String, &'a mut Tensor<'a, T>);
//
//    type IntoIter = std::collections::hash_map::IterMut<'a, String, Tensor<'a, T>>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        (&mut self.parameters).into_iter()
//    }
//}
