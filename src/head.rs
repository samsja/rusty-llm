use crate::utils::fill_tril;
use ndarray::{Array, Ix2, NdFloat};

pub struct Head<T>
where
    T: NdFloat,
{
    w_q: Array<T, Ix2>,
    w_k: Array<T, Ix2>,
    w_v: Array<T, Ix2>,
    dim_key: usize,
}

impl<T> Head<T>
where
    T: NdFloat,
{
    pub fn new_zeros(embed_dim: usize, dim_key: usize, dim_val: usize) -> Head<T> {
        let w_q = Array::<T, _>::zeros((embed_dim, dim_key));
        let w_k = Array::<T, _>::zeros((embed_dim, dim_key));
        let w_v = Array::<T, _>::zeros((embed_dim, dim_val));

        Head {
            w_q,
            w_k,
            w_v,
            dim_key,
        }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        let mut scores = q.dot(&k.t()) / T::from(self.dim_key).unwrap().sqrt();
        let mut mask_scores = fill_tril(&mut scores, T::from(-1e-9).unwrap());
        mask_scores.mapv_inplace(|a| a.exp());

        let output =  mask_scores.clone() / mask_scores.sum();

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

        let head = Head::<f32>::new_zeros(embed_dim, 64, 64);

        head.attention(&embed);
    }
}
