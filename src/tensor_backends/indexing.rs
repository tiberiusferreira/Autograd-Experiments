#[derive(Debug)]
pub struct Indexer{
    original_shape: Vec<usize>,
    current_index: Vec<usize>,
    max_index: Vec<usize>,
    dims: usize,
}

impl From<&[usize]> for Indexer{
    fn from(shape: &[usize]) -> Self {
        assert!(shape.len() > 0, "Indexer source needs at least one dimension");
        let source_vec = shape.to_vec();
        let dims = source_vec.len();
        let mut zeros = vec![];
        for _dim in 0..dims{
            zeros.push(0);
        }
        let mut max_index = vec![];
        for dim in 0..dims{
            max_index.push(shape[dim] - 1);
        }

        Indexer{
            original_shape: source_vec,
            current_index: vec![],
            dims,
            max_index
        }
    }
}

// We may need GAT to implement iterator here
impl Indexer{
    pub fn next(&mut self) -> Option<&[usize]>{
        if self.current_index == self.max_index{
            return None
        }else if self.current_index.is_empty(){
            for dim in 0..self.dims{
                self.current_index.push(0);
            }
            return Some(self.current_index.as_slice());
        }else{
            for dim in (0..self.dims).rev(){
                if self.current_index[dim] < self.original_shape[dim] - 1{
                    self.current_index[dim] += 1;
                    break;
                }else{
                    self.current_index[dim] = 0;
                }
            }
        }
        Some(self.current_index.as_slice())
    }
}

#[cfg(test)]
mod indexing_tests {
    use super::*;
    use crate::tape::*;
    use crate::tensor_backends::NdArray;

    #[test]
    fn index_test() {
        let k: &[usize] = &[2, 3];
        let mut indexer: Indexer = Indexer::from(k);
        assert_eq!(indexer.next(), Some([0usize,0].as_ref()));
        assert_eq!(indexer.next(), Some([0usize,1].as_ref()));
        assert_eq!(indexer.next(), Some([0usize,2].as_ref()));
        assert_eq!(indexer.next(), Some([1usize,0].as_ref()));
        assert_eq!(indexer.next(), Some([1usize,1].as_ref()));
        assert_eq!(indexer.next(), Some([1usize,2].as_ref()));
        assert_eq!(indexer.next(), None);
        assert_eq!(indexer.next(), None);
    }
}

