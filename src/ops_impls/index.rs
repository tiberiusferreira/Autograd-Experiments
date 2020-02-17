use crate::{Op, Tensor, TensorBackend};
use ndarray::prelude::IxDyn;
use std::rc::Rc;
use std::cell::RefCell;
use std::borrow::Borrow;
use std::ops::Deref;

#[derive(Debug)]
pub struct IndexOp<T: TensorBackend> {
    original_shared: Rc<RefCell<Tensor<T>>>,
}

pub fn index<T: TensorBackend>(original: Rc<RefCell<Tensor<T>>>) -> Tensor<T> {
    let index_op: IndexOp<T> = IndexOp {
        original_shared: original
    };
    index_op.forward()
}

impl<T: TensorBackend> Op<T> for IndexOp<T> {
    fn name(&self) -> String {
        "IndexOp".to_string()
    }

    fn forward(self) -> Tensor<T> {
        let result = (&(*self.original_shared).borrow().data).clone();
        Tensor::from_op_result(result, Box::new(self))
    }

    fn set_operand_grad(&mut self, previous_op_grad: Tensor<T>) {
//        self.original.grad = Some(Box::new(previous_op_grad))
        unimplemented!()
    }

    fn operands(&self) -> Vec<&Tensor<T>> {
        //: &Tensor<T>
        vec![&self.original_shared.deref().borrow()]
//        vec![&a]
//        unimplemented!()
//        a
    }

    fn operands_mut(&mut self) -> Vec<&mut Tensor<T>> {
//        vec![&mut self.original.borrow_mut()]
        unimplemented!()
    }

}
