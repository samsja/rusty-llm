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

        let y = (x - x.mean().unwrap()) / (x.var(T::from(1).unwrap()) + eps).sqrt();

        y * &self.weight + &self.bias
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
}
