use crate::float::MyFloat;
use crate::nn::dot::dot_3d_3d;
use crate::nn::linear::Linear;
use crate::nn::utils::{fill_tril_3d, softmax_inplace_3d};
use ndarray::{Array, Axis, Ix2, Slice};

pub struct CausalHead<T>
where
    T: MyFloat,
{
    qkv: Linear<T>, // Q, K, V at the same time
    proj: Linear<T>,
}

impl<T> CausalHead<T>
where
    T: MyFloat,
{
    pub fn new_zeros(embed_dim: usize) -> CausalHead<T> {
        let qkv = Linear::<T>::new_zeros(embed_dim, 3 * embed_dim);

        let proj = Linear::<T>::new_zeros(embed_dim, embed_dim);

        CausalHead::new(qkv, proj)
    }
    pub fn new(qkv: Linear<T>, proj: Linear<T>) -> CausalHead<T> {
        CausalHead { qkv, proj }
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let num_head = 2; // make parameter
        let embed_dim = input.shape()[1];

        let qkv = self.qkv.forward(&input); // (seq, 3* embed) = (seq, embed) @ (embed, 3* embed)

        let q = qkv.slice_axis(Axis(1), Slice::from(..embed_dim));
        let k = qkv.slice_axis(Axis(1), Slice::from(embed_dim..2 * embed_dim));
        let v = qkv.slice_axis(Axis(1), Slice::from(2 * embed_dim..));

        let seq_len = q.shape()[0];

        let q = q
            .into_owned()
            .into_shape((num_head, seq_len, embed_dim / num_head))
            .unwrap();
        let k = k
            .into_owned()
            .into_shape((num_head, embed_dim / num_head, seq_len))
            .unwrap();

        let qk = dot_3d_3d(&q.view(), &k.view());

        let mut scores = qk / T::from(embed_dim).unwrap().sqrt(); // (seq,seq) = (seq, embed) @ (embed, seq)

        let mut mask_scores = fill_tril_3d(&mut scores, T::from(-1e9).unwrap());
        softmax_inplace_3d(&mut mask_scores);

        let v = v
            .into_owned()
            .into_shape((num_head, seq_len, embed_dim / num_head))
            .unwrap();

        let output = dot_3d_3d(&mask_scores.view(), &v.view());

        let output = output.into_shape((seq_len, embed_dim)).unwrap();

        self.proj.forward(&output) // (embed, seq) = (embed, embed) @ (embed, seq )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_attention() {
        let embed_dim = 6;
        let seq_len = 4;

        let embed = Array::<f32, _>::zeros((seq_len, embed_dim).f());

        let head = CausalHead::<f32>::new_zeros(embed_dim);

        head.attention(&embed);
    }

    #[test]
    fn test_attention_exact_forward() {
        let embed = array!(
            [1.3486282, -0.3646578, 0.2202760, -0.0248854],
            [-0.8526461, 0.6483270, 0.7306849, -1.1708095],
            [-0.0693992, -1.7500497, 0.0960634, 0.3073791],
        );

        let weight = array!(
            [
                0.3079206, -0.0546556, -0.4739076, 0.1702384, 0.3945509, -0.4402612, -0.4789515,
                0.4983537, 0.1441541, -0.1878942, -0.4490049, 0.4938900
            ],
            [
                0.2748947, -0.2371672, -0.4005211, 0.4039117, -0.0208597, -0.4016056, 0.2711517,
                0.1972921, -0.2942290, 0.3598832, 0.0386456, -0.0977293
            ],
            [
                0.4967605, -0.0406564, -0.0507813, 0.4248428, 0.2512364, 0.4515253, 0.3624291,
                -0.3666286, -0.0281138, 0.2391329, -0.4096958, -0.4598352
            ],
            [
                0.2609192, -0.1036394, 0.4978991, 0.0982752, 0.4343238, -0.1428618, 0.2879207,
                0.2881057, 0.3257278, -0.2231289, 0.3471265, -0.3942446
            ]
        );

        let bias = array!(
            0.4391516, -0.3696769, 0.3999102, -0.4653839, 0.2069907, -0.4497023, 0.1849731,
            0.3248928, 0.0445384, -0.3840317, -0.1586986, -0.2910188
        );

        let qkv = Linear::<f32>::new(weight, bias);

        let weight = array!(
            [-0.2433862, -0.1706176, 0.0799962, 0.4658914],
            [0.0830147, 0.0612363, 0.2458661, -0.0796878],
            [-0.0991868, 0.3646556, 0.0740875, 0.3419811],
            [-0.0934386, -0.4345958, -0.2550721, 0.4669699]
        );

        let bias = array!(-0.2967021, -0.1021943, -0.4964519, 0.4786599);

        let proj = Linear::<f32>::new(weight, bias);

        let head = CausalHead::<f32>::new(qkv, proj);

        let output = head.attention(&embed);

        assert_eq!(output.mean().unwrap(), -0.1483888);
    }
}
