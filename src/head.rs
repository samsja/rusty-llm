use ndarray::prelude::*;
use ndarray::{Array, Ix2, NdFloat};

pub struct Head<T>
where
    T: NdFloat,
{
    w_q: Array<T, Ix2>,
    w_k: Array<T, Ix2>,
    w_v: Array<T, Ix2>,
    dim_key: usize,
    dim_val: usize,
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
            dim_val,
        }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        let scores = q.dot(&k.t()) / T::from(self.dim_key).unwrap().sqrt();
        let exp_scores = scores.mapv(|a| a.exp());

        let output = exp_scores.clone() / exp_scores.sum();

        output.dot(&v)
    }
}
