use crate::float::MyFloat;
use crate::nn::head::CausalHead;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::linear::Linear;
use ndarray::{Array, Ix2};
use std::f32::consts::PI;

pub fn new_gelu_inplace<'a, T: MyFloat>(x: &'a mut Array<T, Ix2>) {
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
    T: MyFloat,
{
    ln_1: LayerNorm<T>,
    head: CausalHead<T>,
    ln_2: LayerNorm<T>,
    fc: Linear<T>,
    proj: Linear<T>,
}

impl<T> Block<T>
where
    T: MyFloat,
{
    pub fn forward(&self, x: &Array<T, Ix2>) -> Array<T, Ix2> {
        // println!("x = {}", x.mean().unwrap());
        let y = self.ln_1.forward(x);
        // println!("ln_1 = {}", y.mean().unwrap());
        let y = self.head.attention(&y);
        // println!("attn = {}", y.mean().unwrap());
        let x = x + y;
        let x_skip = &x;
        let x = self.ln_2.forward(&x);
        let mut x = self.fc.forward(&x);
        new_gelu_inplace(&mut x);
        let x = self.proj.forward(&x);
        let x = x_skip + x;
        // println!("mlp {}", x.mean().unwrap());
        x
    }

    pub fn new_zeros(embed_dim: usize) -> Block<T> {
        let ln_1 = LayerNorm::<T>::new_zeros(embed_dim);
        let head = CausalHead::<T>::new_zeros(embed_dim);
        let ln_2 = LayerNorm::<T>::new_zeros(embed_dim);
        let fc = Linear::<T>::new_zeros(embed_dim, 4 * embed_dim);
        let proj = Linear::<T>::new_zeros(4 * embed_dim, embed_dim);
        Block::new(ln_1, head, ln_2, fc, proj)
    }

    pub fn new(
        ln_1: LayerNorm<T>,
        head: CausalHead<T>,
        ln_2: LayerNorm<T>,
        fc: Linear<T>,
        proj: Linear<T>,
    ) -> Block<T> {
        Block::<T> {
            ln_1,
            head,
            ln_2,
            fc,
            proj,
        }
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
