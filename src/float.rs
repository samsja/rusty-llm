use half::{bf16, f16};
use ndarray::ScalarOperand;
use num_traits::Float;

pub trait MyFloat: 'static + Float + ScalarOperand {}

impl MyFloat for f32 {}
impl MyFloat for f64 {}
impl MyFloat for f16 {}
impl MyFloat for bf16 {}
