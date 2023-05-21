use crate::float::MyFloat;
use crate::nn::dot::dot_3d_3d_par;
use crate::nn::linear::Linear;
use crate::nn::utils::{fill_tril_3d, softmax_inplace_3d};
use ndarray::{Array, ArrayView, Axis, CowArray, Ix2, Ix3, Slice};

pub struct CausalHead<T>
where
    T: MyFloat,
{
    qkv: Linear<T>, // Q, K, V at the same time
    proj: Linear<T>,
    num_head: usize,
}

impl<T> CausalHead<T>
where
    T: MyFloat,
{
    pub fn new_zeros(embed_dim: usize) -> CausalHead<T> {
        let qkv = Linear::<T>::new_zeros(embed_dim, 3 * embed_dim);

        let proj = Linear::<T>::new_zeros(embed_dim, embed_dim);

        CausalHead::new(qkv, proj, 2)
    }
    pub fn new(qkv: Linear<T>, proj: Linear<T>, num_head: usize) -> CausalHead<T> {
        CausalHead {
            qkv,
            proj,
            num_head,
        }
    }

    fn reshape_m<'a>(&'a self, m: &'a ArrayView<T, Ix2>) -> CowArray<T, Ix3> {
        let embed_dim = m.shape()[1];
        let seq_len = m.shape()[0];

        let mut m = m
            .to_shape((seq_len, self.num_head, embed_dim / self.num_head))
            .unwrap();

        m.swap_axes(0, 1);
        m
    }

    pub fn attention(&self, input: &Array<T, Ix2>) -> Array<T, Ix2> {
        let embed_dim = input.shape()[1];

        let qkv = self.qkv.forward(&input); // (seq, 3* embed) = (seq, embed) @ (embed, 3* embed)

        let q = qkv.slice_axis(Axis(1), Slice::from(..embed_dim));
        let k = qkv.slice_axis(Axis(1), Slice::from(embed_dim..2 * embed_dim));
        let v = qkv.slice_axis(Axis(1), Slice::from(2 * embed_dim..));

        let seq_len = q.shape()[0];

        let q = self.reshape_m(&q);
        let mut k = self.reshape_m(&k);

        k.swap_axes(2, 1);

        let qk = dot_3d_3d_par(&q.view(), &k.view());

        let norm = 1.0 / (k.shape()[1] as f32).sqrt();

        let mut scores = qk * T::from(norm).unwrap();

        let mut mask_scores = fill_tril_3d(&mut scores, T::from(-1e9).unwrap());
        softmax_inplace_3d(&mut mask_scores);

        let v = self.reshape_m(&v);

        let mut output = dot_3d_3d_par(&mask_scores.view(), &v.view());

        output.swap_axes(0, 1);

        let output = output.as_standard_layout();

        let output = output.to_shape((seq_len, embed_dim)).unwrap();

        self.proj.forward_cow(&output) // (embed, seq) = (embed, embed) @ (embed, seq )
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
            [0.4975254, 0.6200650, 0.2310395, 0.7172407, 0.3621508, 0.6562123],
            [0.1848982, 0.7425442, 0.9772386, 0.5010736, 0.2336625, 0.9492494],
            [0.9412946, 0.2011762, 0.0401002, 0.6062413, 0.9571846, 0.9763392],
            [0.9412946, 0.2011762, 0.0401002, 0.6062413, 0.9571846, 0.9763392],
        );

        let weight = array!(
            [0.3953309, -0.3597091, -0.2734019, 0.0251472, 0.0165467, 0.2952037],
            [-0.0267402, 0.1615911, -0.3413630, 0.1260773, 0.2927725, 0.3384781],
            [-0.2275943, 0.0004362, 0.2393830, -0.0366230, 0.1357679, -0.0491278],
            [-0.1999444, -0.0862556, 0.2520943, 0.1719027, -0.2428987, 0.2977147],
            [-0.3208715, 0.3368620, 0.2660686, -0.2797337, -0.2899868, 0.1329844],
            [0.1739980, -0.3727498, -0.2812749, 0.0675823, -0.2737234, 0.0868654],
            [0.0585976, -0.2309992, -0.0093142, -0.3670481, -0.0117402, 0.3975820],
            [0.3122012, -0.0672695, -0.3864970, 0.2636839, 0.0298996, 0.2747818],
            [-0.2932634, 0.1497673, -0.2231018, 0.1706953, 0.0957896, -0.3954427],
            [0.1624354, 0.0266357, 0.3917891, 0.3158199, 0.1244801, -0.1019748],
            [-0.3595042, -0.2926345, 0.2158656, 0.1169033, 0.0837029, -0.0485494],
            [0.1102039, 0.0726328, 0.3436545, -0.1462546, 0.2204990, -0.2131337],
            [0.3384728, -0.0305376, -0.1818183, 0.0047107, -0.2616675, 0.1932835],
            [-0.1481870, -0.0512277, 0.0779617, 0.0897861, 0.2679144, 0.3620642],
            [0.1294264, 0.3414125, 0.3936889, -0.1379894, 0.0691708, -0.3550798],
            [0.3522206, -0.1665217, -0.2154174, 0.3798443, 0.0965101, 0.2447864],
            [0.3805540, -0.1632809, 0.1414836, -0.3316701, 0.1403766, 0.0190480],
            [0.2974199, -0.2874658, 0.0524018, -0.1284005, 0.2180872, 0.0604943],
        );

        let weight = weight.t().to_owned();

        let bias = array!(
            -0.2002964, -0.3890274, -0.0394395, 0.1040076, 0.1079955, -0.3467788, 0.1674609,
            0.1168134, 0.3650604, -0.2834172, -0.3763380, -0.2445963, -0.2156843, 0.1507809,
            -0.1848304, 0.0470366, -0.0399221, 0.2149469,
        );

        let qkv = Linear::<f32>::new(weight, bias);

        let weight = array!(
            [-0.0954513, 0.0803510, -0.3928013, -0.2576592, 0.2471844, 0.3736498],
            [-0.0918460, -0.2167355, -0.0792394, 0.4072537, -0.2233375, -0.1992978],
            [0.2132241, 0.2933911, 0.2147706, -0.3484667, -0.0083705, -0.2828131],
            [-0.4029792, -0.2235603, -0.1888681, 0.1858437, -0.1869524, -0.3629007],
            [-0.3497532, 0.1356127, -0.2289001, 0.1569834, -0.1669361, 0.3261105],
            [-0.2345909, -0.3837104, 0.2967449, 0.2585668, -0.3883521, -0.2670496],
        );
        let weight = weight.t().to_owned();

        let bias = array!(0.0877704, -0.2477426, -0.3182796, 0.0820071, -0.1132862, 0.0899569);

        let linear = Linear::<f32>::new(weight, bias);

        let head = CausalHead::<f32>::new(qkv, linear, 2);

        let output = head.attention(&embed);

        assert_eq!(output.mean().unwrap(), -0.05929202);
    }
}
