use half::{bf16, f16};
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::fmt::Display;

pub trait FromFourBytes {
    fn from_f32_bytes(bytes: [u8; 4]) -> Self;
}

impl FromFourBytes for f32 {
    fn from_f32_bytes(bytes: [u8; 4]) -> f32 {
        f32::from_le_bytes(bytes)
    }
}

impl FromFourBytes for f16 {
    fn from_f32_bytes(bytes: [u8; 4]) -> f16 {
        f16::from_f32(f32::from_le_bytes(bytes))
    }
}

impl FromFourBytes for bf16 {
    fn from_f32_bytes(bytes: [u8; 4]) -> bf16 {
        bf16::from_f32(f32::from_le_bytes(bytes))
    }
}

pub trait MyFloat:
    'static + Float + ScalarOperand + FromFourBytes + FromPrimitive + Display
{
}

impl MyFloat for f32 {}
impl MyFloat for f16 {}
impl MyFloat for bf16 {}
