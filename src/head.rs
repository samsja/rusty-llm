use crate::utils::fill_tril;
use ndarray::{Array, Axis, Ix2, NdFloat, Slice};

pub struct CausalHead<T>
where
    T: NdFloat,
{
    qkv: Array<T, Ix2>, // Q, K, V at the same time
    proj: Array<T, Ix2>,
}

impl<T> CausalHead<T>
where
    T: NdFloat,
{
    pub fn new_zeros(embed_dim: usize) -> CausalHead<T> {
        let qkv = Array::<T, _>::zeros((3 * embed_dim, embed_dim));
        let proj = Array::<T, _>::zeros((embed_dim, embed_dim));

        CausalHead { qkv, proj }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let embed_dim = input.shape()[1];
        let qkv = input.dot(&self.qkv.t()); // (seq, 3* embed) = (seq, embed) @ (embed, 3* embed)

        let q = qkv.slice_axis(Axis(1), Slice::from(..embed_dim));
        let k = qkv.slice_axis(Axis(1), Slice::from(embed_dim..2 * embed_dim));
        let v = qkv.slice_axis(Axis(1), Slice::from(2 * embed_dim..));
        

        let mut scores = q.dot(&k.t()) / T::from(embed_dim).unwrap().sqrt(); // (seq,seq) = (seq, embed) @ (embed, seq)
        let mask_scores = fill_tril(&mut scores, T::from(-1e9).unwrap());
        mask_scores.mapv_inplace(|a| a.exp());

        let output = mask_scores.clone() / mask_scores.sum();

        self.proj.dot(&output.dot(&v).t()) // (embed, seq) = (embed, embed) @ (embed, seq )
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

    #[test]
    fn test_attention_mingpt() {
        let embed = Array::<f32, _>::from(vec![[0.8032, -0.2442], [-0.7848, 0.9664]]);
        let qkv = Array::<f32, _>::from(vec![
            [-0.4762, 0.5736],
            [-0.0200, -0.6306],
            [-0.0960, 0.4737],
            [0.1597, 0.3231],
            [0.2959, -0.6629],
            [0.1549, 0.1660],
        ]);

        let proj = Array::<f32, _>::from(vec![[0.4662, 0.0378], [-0.2131, -0.2160]]);

        let head = CausalHead { qkv, proj };

        let _output = head.attention(&embed);

/*         assert_eq!( */
            // output,
            // Array::<f32, _>::from(vec![[0.8703, -0.3495], [0.5097, -0.1793]]),
        /* ); */
    }
}
