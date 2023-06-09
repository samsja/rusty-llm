use crate::float::MyFloat;
use ndarray::{Array, CowArray, Ix1, Ix2};

pub struct Linear<T>
where
    T: MyFloat,
{
    weight: Array<T, Ix2>,
    bias: Array<T, Ix1>,
}

impl<T> Linear<T>
where
    T: MyFloat,
{
    pub fn forward(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        self.forward_cow(&CowArray::from(input))
    }

    pub fn forward_cow(&self, input: &CowArray<T, Ix2>) -> Array<T, Ix2> {
        let output = input.dot(&self.weight);
        output + &self.bias // todo could be optimize to do inplace
    }

    pub fn new(weight: Array<T, Ix2>, bias: Array<T, Ix1>) -> Linear<T> {
        Linear { weight, bias }
    }

    pub fn new_zeros(dim_in: usize, dim_out: usize) -> Linear<T> {
        let weight = Array::<T, _>::zeros((dim_in, dim_out));
        let bias = Array::<T, _>::zeros(dim_out);

        Linear::<T>::new(weight, bias)
    }
}

pub struct LinearNoBias<T>
where
    T: MyFloat,
{
    weight: Array<T, Ix2>,
}

impl<T> LinearNoBias<T>
where
    T: MyFloat,
{
    pub fn forward(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        input.dot(&self.weight.t()) // contrary to Linear we do a transpose. This is because of gpt2 weight, might change in the future
    }

    pub fn new(weight: Array<T, Ix2>) -> LinearNoBias<T> {
        LinearNoBias { weight }
    }

    pub fn new_zeros(dim_in: usize, dim_out: usize) -> LinearNoBias<T> {
        let weight = Array::<T, _>::zeros((dim_out, dim_in));
        LinearNoBias::<T>::new(weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_forward() {
        let input_dim = 3;
        let output_dim = 4;
        let seq = 2;

        let linear = Linear::<f32>::new_zeros(input_dim, output_dim);

        let input = Array::<f32, _>::zeros((seq, input_dim));

        linear.forward(&input);
    }
    #[test]
    fn test_forward_no_bias() {
        let input_dim = 3;
        let output_dim = 4;
        let seq = 2;

        let linear = LinearNoBias::<f32>::new_zeros(input_dim, output_dim);

        let input = Array::<f32, _>::zeros((seq, input_dim));

        linear.forward(&input);
    }

    #[test]
    fn test_f16() {
        let input_dim = 3;
        let output_dim = 4;
        let seq = 2;

        let linear = Linear::<f16>::new_zeros(input_dim, output_dim);

        let input = Array::<f16, _>::zeros((seq, input_dim));

        linear.forward(&input);
    }
}
