use crate::tensor_backends::TensorBackend;
use crate::Tensor;
use crate::tensor_backends::indexing::Indexer;
/// Here we calculate the output to a given input. Save the output gradient w.r.t. the input
/// Then change the input by delta, rerun the computation and check that the output changed by
/// delta*gradient up to given precision
pub fn validate_grad<T: TensorBackend>(input: Tensor<T>, computation: &dyn for<'b> Fn(&Tensor<'b, T>) -> Tensor<'b, T>){
    // Here we should index each element of the input tensor and verify that its gradient is correct
    // This looks like, for Rank 3: [0, 0, 0], [0, 0, 1], [0, 1, 0] ... and so on.
    let mut indexer = Indexer::from(input.shape());
    let output: Tensor<T> = computation(&input);
    while let Some(i) = indexer.next() {
        println!("verifying index {:?}", i);
        let first_index: &[usize] = &[0usize];
        let output_no_delta = output.data().index(first_index);
        let output_grad_wrt_input = output.grad().wrt(&input).data().index(i);

        let delta = 0.01;
        let mut data_plus_delta = input.data().clone();
        let index_0: &mut f32 = data_plus_delta._index_mut(i);
        *index_0 += delta;

        let input_with_delta = output.tape.new_from_backend_value(data_plus_delta);

        let new_output: Tensor<T> = computation(&input_with_delta);

        let actual_output = new_output.data().index(first_index);
        let predicted_output = output_no_delta + delta * output_grad_wrt_input;
        let error = (actual_output-predicted_output).abs();
        println!("Delta: {:?}", delta);
        println!("Grad: {:?}", output_grad_wrt_input);
        println!("actual: {:?}", actual_output);
        println!("predicted: {:?}", predicted_output);
        assert!(error < 0.001);
    }

}