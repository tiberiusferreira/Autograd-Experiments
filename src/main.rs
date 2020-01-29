use backprop::*;


//struct Model {
//    trainable_vars: Vec<Var>
//}
//impl Model{
//    pub fn new() -> Self{
//        Model{
//            trainable_vars: vec![]
//        }
//    }
//}

//pub fn mul<'mut_ref, 'var>(var1: &'a mut Var<'a>, var2: &'a mut Var<'a>){
//    let c: Var<'a> = var1 * var2;
//}


//pub struct VarStore{
//    vars: Vec<Tensor>
//}

//pub struct VarIndex(usize);
//
//impl VarStore{
//    pub fn new() -> VarStore{
//        VarStore{
//            vars: vec![]
//        }
//    }
//
//    pub fn new_var(&mut self, value: f32) -> VarIndex {
//        let new_var = Tensor;
//        self.vars.push(new_var(value));
//        let len = self.vars.len();
//        VarIndex(len - 1)
//    }
//
//    pub fn var(&self, index: VarIndex) -> &Tensor {
//        &self.vars[index.0]
//    }
//
//    pub fn mul(&mut self, left: VarIndex, right: VarIndex) -> VarIndex{
//        let left = self.var(left);
//        let right = self.var(right);
//        let value = left.0*right.0;
//        self.new_var(value)
//    }
//
//}



fn main() {
    let mut vs: ParameterStore = ParameterStore::new();

    for _i in 0..20 {
        let a = vs.remove_or_init("a".into(), 2.);
        let b = Tensor::new(2.);
        let mut c = a * b;
        c = relu(c);

        vs = c.get_trainable_with_grads();

        for (_id, tensor) in &mut vs {
            tensor.data = tensor.data - tensor.grad.unwrap()*0.1;
            tensor.grad = None;
        }

        println!("{:?}", c.data);
    }

}

//pub fn model() -> Var{
//    let a: Var = Var::new(1.); // grad should be 2
//    let b: Var = Var::new(2.).trainable(true); // grad should be 1
//    let c: Var = a*b;
//    let mut loss: Var = c*Var::new(3.);
//    loss
//}
