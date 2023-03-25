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

        let scores = q.dot(&k.t()) / T::from(self.dim_key).unwrap().sqrt();
        let exp_scores = scores.mapv(|a| a.exp());

        let output = exp_scores.clone() / exp_scores.sum();

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
        println!("{:?}", embed);

        let head = Head::<f32>::new_zeros(embed_dim, 64, 64);

        head.attention(&embed);
    }
}
