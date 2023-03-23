use ndarray::prelude::*;
use ndarray::{Array, Ix2, NdFloat};

pub struct Head<T>
where
    T: NdFloat,
{
    w_Q: Array<T, Ix2>,
    w_K: Array<T, Ix2>,
    w_V: Array<T, Ix2>,
    dim_key: usize,
    dim_val: usize,
}

impl<T> Head<T>
where
    T: NdFloat,
{
    pub fn new_zeros(embed_dim: usize, dim_key: usize, dim_val: usize) -> Head<T> {
        let w_Q = Array::<T, _>::zeros((embed_dim, dim_key));
        let w_K = Array::<T, _>::zeros((embed_dim, dim_key));
        let w_V = Array::<T, _>::zeros((embed_dim, dim_val));

        Head {
            w_Q,
            w_K,
            w_V,
            dim_key,
            dim_val,
        }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let Q = input.dot(&self.w_Q);
        let K = input.dot(&self.w_K);
        let V = input.dot(&self.w_V);

        let scores = Q.dot(&K.t()) / T::from(self.dim_key).unwrap().sqrt();
        let exp_scores = scores.mapv(|a| a.exp());

        let output = exp_scores.clone() / exp_scores.sum();

        output.dot(&V)
    }
}

fn main() {
    let embed_dim = 100;
    let seq_len = 10;

    let embed = Array::<f32, _>::zeros((seq_len, embed_dim).f());
    println!("{:?}", embed);

    let head = Head::<f32>::new_zeros(embed_dim, 64, 64);

    let att = head.attention(&embed);

    println!("{:?}", att);
}
