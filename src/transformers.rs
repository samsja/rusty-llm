use crate::utils::fill_tril;
use ndarray::{Array, Ix2, NdFloat};

pub struct CausalHead<T>
where
    T: NdFloat,
{
    w_q: Array<T, Ix2>, // TODO: optimization the three matrix should be contigious
    w_k: Array<T, Ix2>,
    w_v: Array<T, Ix2>,
}

impl<T> CausalHead<T>
where
    T: NdFloat,
{
    pub fn new_zeros(embed_dim: usize) -> CausalHead<T> {
        let w_q = Array::<T, _>::zeros((embed_dim, embed_dim));
        let w_k = Array::<T, _>::zeros((embed_dim, embed_dim));
        let w_v = Array::<T, _>::zeros((embed_dim, embed_dim));

        CausalHead {
            w_q,
            w_k,
            w_v,
        }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        let mut scores = q.dot(&k.t()) / T::from(self.w_v.shape()[0]).unwrap().sqrt();
        let mask_scores = fill_tril(&mut scores, T::from(-1e9).unwrap());
        mask_scores.mapv_inplace(|a| a.exp());

        let output = mask_scores.clone() / mask_scores.sum();

        output.dot(&v)
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
