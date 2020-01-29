pub fn assert_very_close_in_value(first: f32, second: f32){
    assert!((first - second).abs() < 1e-4, "first = {} second = {}", first, second);
}