use crate::nn::linear::Linear;
use crate::nn::utils::fill_tril;
use ndarray::{Array, Axis, Ix2, NdFloat, Slice};

pub struct CausalHead<T>
where
    T: NdFloat,
{
    qkv: Linear<T>, // Q, K, V at the same time
    proj: Linear<T>,
}

impl<T> CausalHead<T>
where
    T: NdFloat,
{
    pub fn new_zeros(embed_dim: usize) -> CausalHead<T> {
        let qkv = Linear::<T>::new_zeros(embed_dim, 3 * embed_dim);

        let proj = Linear::<T>::new_zeros(embed_dim, embed_dim);

        CausalHead { qkv, proj }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let embed_dim = input.shape()[1];

        let qkv = self.qkv.forward(&input); // (seq, 3* embed) = (seq, embed) @ (embed, 3* embed)

        let q = qkv.slice_axis(Axis(1), Slice::from(..embed_dim));
        let k = qkv.slice_axis(Axis(1), Slice::from(embed_dim..2 * embed_dim));
        let v = qkv.slice_axis(Axis(1), Slice::from(2 * embed_dim..));

        let mut scores = q.dot(&k.t()) / T::from(embed_dim).unwrap().sqrt(); // (seq,seq) = (seq, embed) @ (embed, seq)
        let mask_scores = fill_tril(&mut scores, T::from(-1e9).unwrap());
        mask_scores.mapv_inplace(|a| a.exp());

        let output = mask_scores.clone() / mask_scores.sum();
        let output = output.dot(&v);
        self.proj.forward(&output) // (embed, seq) = (embed, embed) @ (embed, seq )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_attention() {
        let embed_dim = 100;
        let seq_len = 10;

        let embed = Array::<f32, _>::zeros((seq_len, embed_dim).f());

        let head = CausalHead::<f32>::new_zeros(embed_dim);

        head.attention(&embed);
    }
}
