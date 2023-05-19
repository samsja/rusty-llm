use crate::float::MyFloat;
use ndarray::{Array, Axis, Ix1, Ix2};

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
        let mut var = x.var_axis(Axis(1), T::from(0).unwrap());
        var.mapv_inplace(|v| (v + eps).sqrt());

        let mean = x.mean_axis(Axis(1)).unwrap();

        let mean = mean.insert_axis(Axis(1)); // add dim 1
        let mean = mean
            .broadcast(x.shape())
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap(); // repeat along dim 1
        let x = x - &mean;

        let var = var.insert_axis(Axis(1)); // add dim 1
        let var = var
            .broadcast(x.shape())
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap(); // repeat along dim 1

        let y = x / var;

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

    #[test]
    fn test_exact_forward_2() {
        let embed = array!(
            [
                0.6838375, 0.3786027, 0.6163493, 0.1548221, 0.2080919, 0.7964638, 0.1401155,
                0.2770312, 0.4225845, 0.9923908
            ],
            [
                0.0241743, 0.9826818, 0.6001092, 0.6853017, 0.8910903, 0.3406627, 0.9750189,
                0.4443324, 0.1682504, 0.6522445
            ]
        );

        let weight = array!(
            0.1279535, 0.3682488, 0.2132850, 0.2362493, 0.0987864, 0.8993521, 0.2769766, 0.2983191,
            0.1761575, 0.0895381
        );

        let bias = array!(
            0.7161613, 0.3097488, 0.5664809, 0.0588822, 0.0648879, 0.5942912, 0.7873974, 0.0160320,
            0.5194538, 0.9585728
        );

        let ln = LayerNorm::<f32>::new(weight, bias);

        let output = ln.forward(&embed);

        let expected = array!(
            [
                0.8160551, 0.1924936, 0.6811613, -0.2067148, -0.0272210, 1.6611562, 0.4613461,
                -0.1880664, 0.4912616, 1.1279584
            ],
            [
                0.4911214, 0.7862722, 0.5825956, 0.1408342, 0.1639027, -0.0809126, 1.1390526,
                -0.1094366, 0.2904685, 0.9802055
            ]
        );

        assert_eq!(output.mean().unwrap() - 3e-8, expected.mean().unwrap());
    }
}
