use crate::float::MyFloat;
use ndarray::{Array, Ix1, Ix2};

pub struct LayerNorm<T>
where
    T: MyFloat,
{
    weight: Array<T, Ix1>,
    bias: Array<T, Ix1>,
}

impl<T> LayerNorm<T>
where
    T: MyFloat,
{
    pub fn new(weight: Array<T, Ix1>, bias: Array<T, Ix1>) -> LayerNorm<T> {
        LayerNorm { weight, bias }
    }

    pub fn forward(&self, x: &Array<T, Ix2>) -> Array<T, Ix2> {
        let eps = T::from(1e-5).unwrap();
        let bottom = (x.var(T::from(0).unwrap()) + eps).sqrt();
        let y = (x - x.mean().unwrap()) / bottom;
        y * &self.weight + &self.bias
    }

    pub fn new_zeros(embed_dim: usize) -> LayerNorm<T> {
        let weight = Array::<T, _>::zeros(embed_dim);
        let bias = Array::<T, _>::zeros(embed_dim);

        LayerNorm::<T>::new(weight, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_linear_layer() {
        let seq_len = 4;
        let embed_dim = 3;

        let embed = Array::<f32, _>::zeros((seq_len, embed_dim).f());

        let weight = Array::<f32, _>::zeros(embed_dim);
        let bias = Array::<f32, _>::zeros(embed_dim);

        let ln = LayerNorm::<f32>::new(weight, bias);

        ln.forward(&embed);
    }

    #[test]
    fn test_exact_forward() {
        let embed = array!([3.0, 4.0, 5.0]);

        let weight = array!(1.0, 2.0, 3.0);
        let bias = array!(1.0, 3.0, 4.0);

        let ln = LayerNorm::<f32>::new(weight, bias);

        let output = ln.forward(&embed);

        let expected = array!([-0.22473562, 3.0000000, 7.6742067]);

        assert_eq!(output, expected);
    }
}
