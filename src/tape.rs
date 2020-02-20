use std::cell::RefCell;
use std::fmt::{Formatter, Error};

/// Tape holds a list of Nodes.
/// Each Node has a corresponding Var somewhere holding an immutable reference to Tape
/// and an index to one of the Nodes. So a Var can get to its Node.
/// The Node for a given Var stores the index and gradient of all direct parents of Var.
/// The Var knows the gradient of its parents because at its creation it has access to the
/// operation which created it.
///
/// Example, for a = x * y
///
/// Node {
///     parents_indices_n_grad: [
///         (x.index, y.value), // ∂a/∂x = y.value
///         (y.index, x.value)  // ∂a/∂y = x.value
///     ]
/// }
///
/// It is designed this way so the gradients from a given Var x backwards can be calculated easily.
/// The final gradients of x's parents are given by the gradient of x times x's parent's gradients.
/// For the example above:
/// Grad x = gradient of a (1) * y.value
/// Grad y = gradient of a (1) * x.value

#[derive(Debug)]
pub struct Tape {
    vars_backwards_data: RefCell<Vec<VarBackwardData>>
}

// #[derive(Debug)]
// struct Node {
//     /// parent index and its gradient
//     parents_indices_n_grad: Vec<(usize, GradFn)>,
// }

#[derive(Debug)]
struct VarBackwardData{
    var_operands_grad_blueprint: Vec<OperandGradBlueprint>
}


/// First argument is the child_grad, second is the current "parent" grad
pub struct GradFn(Box<dyn Fn(Vec<f64>, &mut Vec<f64>)>);

impl std::fmt::Debug for GradFn{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_str("GradFn")
    }
}



#[derive(Debug)]
struct OperandGradBlueprint {
    /// The index where to store the gradient calculated by the grad_fn in the output gradient
    /// structure. The gradient calculated is the gradient of the operand in question
    /// the Var associated with this struct.
    /// This index is the same as the one inside the parent Var.
    operand_index: usize,
    /// Used to initialize the gradients (to zero)
    grad_shape: usize,
    /// Function which takes the current Var gradient and a mutable reference to the current gradient
    /// of one of the operands of the operation that resulted in this Var
    grad_fn: GradFn
}



#[derive(Debug)]
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
    value: Vec<f64>,
}

impl Tape {
    pub fn new() -> Self {
        Tape { vars_backwards_data: RefCell::new(Vec::new()) }
    }

    //noinspection RsNeedlessLifetimes
    pub fn new_var<'t>(&'t self, value: &[f64]) -> Var<'t> {
        Var {
            tape: self,
            value: value.to_vec(),
            index: self.push_new_var_get_index(),
        }
    }

    pub fn len(&self) -> usize {
        self.vars_backwards_data.borrow().len()
    }

    fn push_new_var_get_index(&self) -> usize {
        let mut nodes = self.vars_backwards_data.borrow_mut();
        let len = nodes.len();
        nodes.push(VarBackwardData{
            var_operands_grad_blueprint: vec![],
        });
        len
    }


    fn push_var_with_parents_get_index(&self, grad_info: Vec<OperandGradBlueprint>) -> usize {
        let mut nodes = self.vars_backwards_data.borrow_mut();
        let len = nodes.len();
        nodes.push(VarBackwardData{
            var_operands_grad_blueprint: grad_info,
        });
        len
    }
}



pub fn mul_vec_f64(left: &Vec<f64>, right: &Vec<f64>) -> Vec<f64>{
    assert_eq!(left.len(), right.len(), "right and left lens not equal");
    let len = left.len();
    let mut result = Vec::with_capacity(len);
    for i in 0..left.len(){
        result.push(left[i] *  right[i]);
    }
    result
}



#[derive(Debug)]
pub struct Grad { all_grads: Vec<Vec<f64>> }

impl Grad {

    //noinspection RsNeedlessLifetimes
    pub fn wrt<'t>(&self, var: &Var<'t>) -> Vec<f64> {
        match self.all_grads.get(var.index){
            None => {
                panic!("This var is not part of the computational graph. Maybe it was created using another Tape");
            },
            Some(grad) => {grad.clone()},
        }
    }
}


impl<'t> Var<'t> {
    pub fn value(&self) -> &Vec<f64> {
        &self.value
    }

    pub fn grad(&self) -> Grad {
        let len = self.tape.len();
        let nodes = self.tape.vars_backwards_data.borrow();
        let mut all_grads: Vec<Vec<f64>> = vec![vec![]; len];
        all_grads[self.index] = vec![1.0];
        for i in (0 .. len).rev() {
            let node = &nodes[i];
            let child_grad = all_grads[i].clone();
            for j in 0..node.var_operands_grad_blueprint.len() {
                let grad_fn = &node.var_operands_grad_blueprint[j].grad_fn;
                let curr_grad = &mut all_grads[node.var_operands_grad_blueprint[j].operand_index];
                // Need to check if curr_grad is empty
                if curr_grad.is_empty(){
                    let new_grad_shape  = node.var_operands_grad_blueprint[j].grad_shape;
                    let mut new_grad = Vec::with_capacity(new_grad_shape);
                    for _i in 0..new_grad_shape {
                        new_grad.push(0.);
                    }
                    *curr_grad = new_grad;
                }
                grad_fn.0(child_grad.clone(), curr_grad);
            }
        }
        Grad { all_grads }
        // unimplemented!()
    }

    pub fn mul(&self, other: &Var<'t>) -> Var<'t>{
        // let parents = vec![(self.index, other.value.clone()), (other.index, self.value.clone())];
        let right_val = other.value.clone();
        let grad_fn_left: GradFn  = GradFn(Box::new(move |child_grad: Vec<f64>, self_grad: &mut Vec<f64>|{
            *self_grad = mul_vec_f64(&right_val, &child_grad);
        }));

        let left_val = self.value.clone();
        let grad_fn_right: GradFn  = GradFn(Box::new(move |child_grad: Vec<f64>, self_grad: &mut Vec<f64>|{
            *self_grad = mul_vec_f64(&left_val, &child_grad);
        }));

        let left_blueprint = OperandGradBlueprint{
            operand_index: self.index,
            grad_shape: self.value.len(),
            grad_fn: grad_fn_left
        };
        let right_blueprint = OperandGradBlueprint{
            operand_index: other.index,
            grad_shape: self.value.len(),
            grad_fn: grad_fn_right
        };
        let parents = vec![left_blueprint, right_blueprint];
        Var {
            tape: self.tape,
            value: mul_vec_f64(&self.value, other.value()),
            index: self.tape.push_var_with_parents_get_index(parents),
        }
        // unimplemented!()
    }

    pub fn index(&self, index: usize) -> Var<'t>{
        let self_len = self.value.len();
        let grad_fn: GradFn = GradFn(Box::new(move |child_grad: Vec<f64>, self_grad: &mut Vec<f64>|{
            // if self_grad.is_empty(){
            //     let mut new_self_grad: Vec<f64> = Vec::with_capacity(self_len);
            //     for _i in 0..self_len{
            //         new_self_grad.push(0.);
            //     }
            //     *self_grad = new_self_grad;
            // }
            self_grad[index] += child_grad[0];
        }));
        let operand_blueprint = OperandGradBlueprint{
            operand_index: self.index,
            grad_shape: self.value.len(),
            grad_fn
        };

        let parent = vec![operand_blueprint];
        Var {
            tape: self.tape,
            value: vec![self.value[index]],
            index: self.tape.push_var_with_parents_get_index(parent),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::Tape;

    #[test]
    fn x_times_y_plus_sin_x() {

        let mut x_grad = 0.;
        let mut x_val = 0.5;
        for _i in 0..1 {
            let t = Tape::new();
            let x = t.new_var(&[1., 2.]);
            let y = t.new_var(&[3., 4.]);
            let z = x.mul(&y);
            let z_0 = z.index(1);
            let grad = z_0.grad();

            // x_val = x_val - grad.wrt(&x)[0];
            // println!("{:#?}", grad.wrt(&y));
            println!("{:#?}", grad);
            // println!("{:#?}", x.value);
            // println!("{:#?}", w.wrt(&x));
            // println!("{:#?}", w.wrt(&y));
            // println!("{:#?}", w.wrt(&z));
        }

    }
}