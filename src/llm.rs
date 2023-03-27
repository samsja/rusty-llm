// decoder only LLM
use crate::head::CausalHead;
use crate::linear::Linear;
use ndarray::{Array, Ix2, NdFloat};
use std::f32::consts::PI;

pub fn new_gelu_inplace<'a, T: NdFloat>(x: &'a mut Array<T, Ix2>) {
    x.mapv_inplace(|v| {
        T::from(0.5).unwrap()
            * v
            * (T::from(1.0).unwrap()
                + ((T::from(2.0 / PI).unwrap().sqrt()
                    * (v + T::from(0.044715).unwrap() * v.powi(3)))
                .tanh()))
    });
}

pub struct Block<T>
where
    T: NdFloat,
{
    head: CausalHead<T>,
    fc: Linear<T>,
    proj: Linear<T>,
}

impl<T> Block<T>
where
    T: NdFloat,
{
    pub fn forward(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let output = self.head.attention(input);
        let output = self.fc.forward(&output);
        let mut output = self.proj.forward(&output);
        new_gelu_inplace(&mut output);

        output
    }

    pub fn new_zeros(embed_dim: usize) -> Block<T> {
        let head = CausalHead::<T>::new_zeros(embed_dim);
        let fc = Linear::<T>::new_zeros(embed_dim, 4 * embed_dim);
        let proj = Linear::<T>::new_zeros(4 * embed_dim, embed_dim);

        Block { head, fc, proj }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_block() {
        let embed_dim = 100;
        let seq_len = 10;

        let embed = Array::<f32, _>::zeros((seq_len, embed_dim).f());

        let block = Block::<f32>::new_zeros(100);

        block.forward(&embed);
    }
}
