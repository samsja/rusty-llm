use half::{bf16, f16};
use ndarray::ScalarOperand;
use num_traits::Float;

struct Myf16(f16);
struct Mybf16(bf16);

impl ScalarOperand for Myf16 {}
impl ScalarOperand for Mybf16 {}



pub trait MyFloat: 'static + Float + ScalarOperand {}

impl MyFloat for f32 {}
impl MyFloat for f64 {}
impl MyFloat for f16 {}
impl MyFloat for bf16 {}
