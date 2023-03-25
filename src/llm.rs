// decoder only LLM
use crate::head::CausalHead;
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
    fc: Array<T, Ix2>,
    proj: Array<T, Ix2>,
}

impl<T> Block<T>
where
    T: NdFloat,
{
    pub fn forward(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let output = self.head.attention(input);
        let output = output.t().dot(&self.fc);
        let mut output = output.dot(&self.proj);
        new_gelu_inplace(&mut output);

        output
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

        let head = CausalHead::<f32>::new_zeros(embed_dim);
        let fc = Array::<f32, _>::zeros((embed_dim, 4 * embed_dim));
        let proj = Array::<f32, _>::zeros((4 * embed_dim, embed_dim));

        let block = Block { head, fc, proj };

        block.forward(&embed);
    }
}
