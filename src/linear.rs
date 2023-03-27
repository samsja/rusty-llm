use ndarray::{Array, Ix1, Ix2, NdFloat};

pub struct Linear<T>
where
    T: NdFloat,
{
    weight: Array<T, Ix2>,
    bias: Array<T, Ix1>,
}

impl<T> Linear<T>
where
    T: NdFloat,
{
    pub fn forward(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let output = input.dot(&self.weight.t());
        print!("dot {:?} \n", output);
        output + self.bias.clone() // could be optimize to do inplace
    }

    pub fn new(weight: Array<T, Ix2>, bias: Array<T, Ix1>) -> Linear<T> {
        if weight.shape()[0] != bias.shape()[0] {
            panic!(
                "Linear layer init: weight of shape {:?} is not compatible with bias of shape {:?}",
                weight.shape(),
                bias.shape()
            );
        }

        Linear { weight, bias }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_forward() {
        let input_dim = 3;
        let output_dim = 4;
        let seq = 2;

        let weight = Array::<f32, _>::zeros((output_dim, input_dim));
        let bias = Array::<f32, _>::zeros(output_dim);

        let linear = Linear::new(weight, bias);

        let input = Array::<f32, _>::zeros((seq, input_dim));

        linear.forward(&input);
    }
}
