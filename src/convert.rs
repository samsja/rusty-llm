// use crate::float::MyFloat;
use ndarray::{Array, Ix1};
use safetensors::tensor::TensorView;

pub fn to_f32<'data>(view: TensorView<'data>) -> Vec<f32> {
    // inspire from smelte-rs.
    // https://github.com/Narsil/smelte-rs/blob/d81f714abce2e64539d3c87dfc6c5488e6a65c03/examples/bert.rs#L168
    // TODO understand what the fuck is going on here. I want to move on towards the end of
    // the project and then deal with this part

    let v = view.data();

    let mut c = Vec::with_capacity(v.len() / 4);
    let mut i = 0;
    while i < v.len() {
        c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
        i += 4;
    }
    c
}

pub fn from_safe_tensorview(tensor: TensorView) -> Array<f32, Ix1> {
    Array::<f32, Ix1>::from(to_f32(tensor))
}
