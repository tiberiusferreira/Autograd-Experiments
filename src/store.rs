use std::collections::HashMap;
use crate::Tensor;

#[derive(Debug)]
pub struct ParameterStore{
    parameters: HashMap<String, Tensor>
}


impl ParameterStore{
    pub fn insert(&mut self, id: String, t: Tensor){
        self.parameters.insert(id, t);
    }
    pub fn get(&mut self, id: &str) -> Option<&Tensor>{
        self.parameters.get(id)
    }
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Tensor>{
        self.parameters.get_mut(id)
    }
    pub fn remove(&mut self, id: &str) -> Option<Tensor>{
        self.parameters.remove(id)
    }
    pub fn remove_or_init(&mut self, id: &str, init_val: f32) -> Tensor{
        self.parameters.remove(id).unwrap_or(Tensor::new_trainable(init_val, id.to_string()))
    }
    pub fn new() -> Self{
        ParameterStore{
            parameters: Default::default()
        }
    }
    pub fn from_hashmap(hash: HashMap<String, Tensor>) -> Self{
        ParameterStore{
            parameters: hash
        }
    }
    pub fn into_raw(self) -> HashMap<String, Tensor>{
        self.parameters
    }


}

impl IntoIterator for ParameterStore{
    type Item = (String, Tensor);

    type IntoIter = std::collections::hash_map::IntoIter<String, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.into_iter()
    }
}


impl<'a> IntoIterator for &'a ParameterStore{
    type Item = (&'a String, &'a Tensor);

    type IntoIter = std::collections::hash_map::Iter<'a, String, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.parameters).into_iter()
    }
}

impl<'a> IntoIterator for &'a mut ParameterStore{
    type Item = (&'a String, &'a mut Tensor);

    type IntoIter = std::collections::hash_map::IterMut<'a, String, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.parameters).into_iter()
    }
}