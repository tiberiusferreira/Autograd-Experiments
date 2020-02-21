use std::cell::RefCell;
use std::fmt::{Formatter, Error};


#[derive(Debug)]
pub struct Tape {
    /// Stores the information necessary to calculate the gradient of the operands of Variables
    /// which reference this Tape. Each Var stores an Index into this structure.
    /// So, in order to construct the gradients we start with a Var and its given gradient,
    /// typically 1 if it stores a single value.
    /// Then we calculate its parents gradients using the data stored in its node, then for each
    /// of those parents we calculate their gradient and so on.
    backwards_data_store: RefCell<Vec<VarBackwardData>>
}

#[derive(Debug)]
struct VarBackwardData{
    /// Stores the data necessary to calculate the gradient of each of the operands of the Op
    /// which created this Var. If the Var was user create, this is empty
    var_operands_grad_blueprint: Vec<OperandGradBlueprint>
}

impl VarBackwardData{
    pub fn empty() -> Self{
        Self{
            var_operands_grad_blueprint: vec![]
        }
    }

    pub fn from_blueprints(blueprints: Vec<OperandGradBlueprint>) -> Self{
        Self{
            var_operands_grad_blueprint: blueprints
        }
    }
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
    operand_tape_index: usize,
    /// Used to initialize the gradients (to zero)
    grad_shape: usize,
    /// Function which takes the current Var gradient and a mutable reference to the current gradient
    /// of one of the operands of the operation that resulted in this Var
    grad_fn: GradFn
}


#[derive(Debug)]
pub struct Var<'t> {
    /// Reference to the Tape which stores the information needed to calculate the gradients
    tape: &'t Tape,
    /// Index of the slot in the tape where the information to calculate the gradient of the
    /// "parents" of this Var are stored
    tape_index: usize,
    /// The actual value of this Var
    value: Vec<f64>,
}

impl Tape {
    pub fn new() -> Self {
        Tape { backwards_data_store: RefCell::new(Vec::new()) }
    }

    //noinspection RsNeedlessLifetimes
    pub fn new_var<'t>(&'t self, value: &[f64]) -> Var<'t> {
        Var {
            tape: self,
            value: value.to_vec(),
            tape_index: self.push(VarBackwardData::empty()),
        }
    }

    pub fn len(&self) -> usize {
        self.backwards_data_store.borrow().len()
    }


    fn push(&self, backwards_data: VarBackwardData) -> usize {
        let mut backwards_data_store = self.backwards_data_store.borrow_mut();
        let len = backwards_data_store.len();
        backwards_data_store.push(backwards_data);
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
        match self.all_grads.get(var.tape_index){
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
        let tape_len = self.tape.len();
        let backwards_data_vec = self.tape.backwards_data_store.borrow();

        let mut all_grads: Vec<Vec<f64>> = vec![vec![]; tape_len];
        // Set self gradient as 1.0
        all_grads[self.tape_index] = vec![1.0];

        let mut tape_indices_to_visit = vec![self.tape_index];

        while let Some(current_tape_index) = tape_indices_to_visit.pop(){
            // Get the data to calculate current Var parents gradients
            let backwards_data = &backwards_data_vec[current_tape_index];
            // Get current Var gradient
            let current_var_grad = all_grads[current_tape_index].clone();
            // For each parent of this Var
            for operand in &backwards_data.var_operands_grad_blueprint {
                // Make sure to visit parents later
                tape_indices_to_visit.push(operand.operand_tape_index);
                // Get the function to calculate the gradient
                let grad_fn = &operand.grad_fn;
                // Get the current gradient
                let curr_grad = &mut all_grads[operand.operand_tape_index];
                // If empty, initialize it
                if curr_grad.is_empty(){
                    let new_grad_shape  = operand.grad_shape;
                    let mut new_grad = Vec::with_capacity(new_grad_shape);
                    for _i in 0..new_grad_shape {
                        new_grad.push(0.);
                    }
                    *curr_grad = new_grad;
                }
                // Update its gradient
                grad_fn.0(current_var_grad.clone(), curr_grad);
            }
        }
        // for tape_index in (0 ..tape_len).rev() {
        //     // Get the data to calculate current Var parents gradients
        //     let backwards_data = &backwards_data_vec[tape_index];
        //     // Get current Var gradient
        //     let current_var_grad = all_grads[tape_index].clone();
        //     // For each parent of this Var
        //     for operand_index in 0..backwards_data.var_operands_grad_blueprint.len() {
        //         // Get the function to calculate the gradient
        //         let grad_fn = &backwards_data.var_operands_grad_blueprint[operand_index].grad_fn;
        //         // Get the current gradient
        //         let curr_grad = &mut all_grads[backwards_data.var_operands_grad_blueprint[operand_index].operand_tape_index];
        //         // If empty, initialize it
        //         if curr_grad.is_empty(){
        //             let new_grad_shape  = backwards_data.var_operands_grad_blueprint[operand_index].grad_shape;
        //             let mut new_grad = Vec::with_capacity(new_grad_shape);
        //             for _i in 0..new_grad_shape {
        //                 new_grad.push(0.);
        //             }
        //             *curr_grad = new_grad;
        //         }
        //         // Update its gradient
        //         grad_fn.0(current_var_grad.clone(), curr_grad);
        //     }
        // }
        Grad { all_grads }
    }

    //noinspection DuplicatedCode
    pub fn mul(&self, other: &Var<'t>) -> Var<'t>{

        let right_val = other.value.clone();
        let grad_fn_left: GradFn  = GradFn(Box::new(move |child_grad: Vec<f64>, self_grad: &mut Vec<f64>|{
            *self_grad = mul_vec_f64(&right_val, &child_grad);
        }));

        let left_val = self.value.clone();
        let grad_fn_right: GradFn  = GradFn(Box::new(move |child_grad: Vec<f64>, self_grad: &mut Vec<f64>|{
            *self_grad = mul_vec_f64(&left_val, &child_grad);
        }));


        let left_blueprint = self.self_gradient_blueprint(grad_fn_left);
        let right_blueprint = other.self_gradient_blueprint(grad_fn_right);

        let var_backwards_data = VarBackwardData::from_blueprints(vec![left_blueprint, right_blueprint]);

        Var {
            tape: self.tape,
            value: mul_vec_f64(&self.value, other.value()),
            tape_index: self.tape.push(var_backwards_data),
        }
    }

    fn self_gradient_blueprint(&self, grad_fn: GradFn) -> OperandGradBlueprint{
        OperandGradBlueprint{
            operand_tape_index: self.tape_index,
            grad_shape: self.value.len(),
            grad_fn
        }
    }


    pub fn index(&self, index: usize) -> Var<'t>{
        let grad_fn: GradFn = GradFn(Box::new(move |child_grad: Vec<f64>, self_grad: &mut Vec<f64>|{
            self_grad[index] += child_grad[0];
        }));

        let operand_blueprint = self.self_gradient_blueprint(grad_fn);

        let blueprints = vec![operand_blueprint];

        let backwards_data = VarBackwardData::from_blueprints(blueprints);
        Var {
            tape: self.tape,
            value: vec![self.value[index]],
            tape_index: self.tape.push(backwards_data),
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