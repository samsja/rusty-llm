use crate::float::MyFloat;
use ndarray::{Array, Ix1, Ix2};
use safetensors::tensor::TensorView;

pub fn to_float<'data, T: MyFloat>(view: &TensorView<'data>) -> Vec<T> {
    // inspire from smelte-rs.
    // https://github.com/Narsil/smelte-rs/blob/d81f714abce2e64539d3c87dfc6c5488e6a65c03/examples/bert.rs#L168
    // TODO understand what the fuck is going on here. I want to move on towards the end of
    // the project and then deal with this part

    let v = view.data();

    let mut c = Vec::with_capacity(v.len() / 4);
    let mut i = 0;
    while i < v.len() {
        c.push(T::from_f32_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
        i += 4;
    }
    c
}

pub fn from_safe_tensorview_1d<T: MyFloat>(tensor: TensorView) -> Array<T, Ix1> {
    Array::<T, Ix1>::from(to_float::<T>(&tensor))
}

pub fn from_safe_tensorview<T: MyFloat>(tensor: TensorView) -> Array<T, Ix2> {
    let shape = tensor.shape();
    Array::<T, Ix1>::from(to_float::<T>(&tensor))
        .into_shape(shape)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap()
}
