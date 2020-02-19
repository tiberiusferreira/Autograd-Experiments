use std::cell::RefCell;

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
pub struct Tape { nodes: RefCell<Vec<Node>> }

enum A{
    Simple(usize, Vec<f64>),
    Mapped(usize, (f64, f64)),
}
#[derive(Debug, Clone)]
struct Node {
    parents_indices_n_grad: Vec<(usize, Vec<f64>)>,
}

#[derive(Debug, Clone)]
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
    value: Vec<f64>,
}

impl Tape {
    pub fn new() -> Self {
        Tape { nodes: RefCell::new(Vec::new()) }
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
        self.nodes.borrow().len()
    }

    fn push_new_var_get_index(&self) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node {
            parents_indices_n_grad: vec![],
        });
        len
    }

    fn push_var_with_parents_get_index(&self, parents: Vec<(usize, Vec<f64>)>) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node {
            parents_indices_n_grad: parents,
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
        let nodes = self.tape.nodes.borrow();
        let mut all_grads: Vec<Vec<f64>> = vec![vec![]; len];
        all_grads[self.index] = vec![1.0];
        for i in (0 .. len).rev() {
            let node = &nodes[i];
            let child_grad = all_grads[i].clone();
            for j in 0..node.parents_indices_n_grad.len() {
                all_grads[node.parents_indices_n_grad[j].0] = mul_vec_f64(&node.parents_indices_n_grad[j].1, &child_grad);
            }
        }
        Grad { all_grads }
    }

    pub fn mul(&self, other: &Var<'t>) -> Var<'t>{
        let parents = vec![(self.index, other.value.clone()), (other.index, self.value.clone())];
        Var {
            tape: self.tape,
            value: mul_vec_f64(&self.value, other.value()),
            index: self.tape.push_var_with_parents_get_index(parents),
        }
    }

    pub fn index(&self, index: usize) -> Var<'t>{
        let parent = vec![(self.index, vec![1.])];
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
        for _i in 0..20 {
            let t = Tape::new();
            let x = t.new_var(&[x_val]);
            let y = t.new_var(&[4.2]);
            let z = x.mul(&y);
            let grad = z.grad();

            x_val = x_val - grad.wrt(&x)[0];
            println!("{:#?}", x.value);
            // println!("{:#?}", w.wrt(&x));
            // println!("{:#?}", w.wrt(&y));
            // println!("{:#?}", w.wrt(&z));
        }

    }
}