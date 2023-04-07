use crate::float::MyFloat;
use ndarray::{Array, ArrayView, Ix1, Ix2};

pub fn fill_tril<'a, T: MyFloat>(x: &'a mut Array<T, Ix2>, val: T) -> &'a mut Array<T, Ix2> {
    // similar to numpy or torch tril
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            if j > i {
                x[[i, j]] = val;
            }
        }
    }

    x
}

pub fn tril<'a, T: MyFloat>(x: &'a mut Array<T, Ix2>) -> &'a Array<T, Ix2> {
    fill_tril(x, T::from(0.0).unwrap())
}

pub fn softmax<T: MyFloat>(x: &ArrayView<T, Ix1>) -> Array<T, Ix1> {
    let max_ = max_val(x);

    let exp_x = x.mapv(|x| (x - max_).exp());

    let sum = exp_x.sum();
    exp_x / sum
}

pub fn max<T: MyFloat>(x: &ArrayView<T, Ix1>) -> (usize, T) {
    let (index, max_val) = x
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (index, *max_val)
}

pub fn argmax<T: MyFloat>(x: &ArrayView<T, Ix1>) -> usize {
    let (index, _) = max(x);
    index
}

pub fn max_val<T: MyFloat>(x: &ArrayView<T, Ix1>) -> T {
    let (_, val) = max(x);
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_trill() {
        let mut mat1 = Array::<f32, _>::ones((3, 3).f());
        let tril_mat = tril(&mut mat1);

        assert_eq!(
            tril_mat,
            Array::<f32, _>::from(vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        );
    }

    #[test]
    fn test_argmax() {
        let index = Array::<f32, _>::from(vec![1.0, 2.0, 3.0, 12.0, 1.0]);
        let max_index = argmax(&index.view());
        assert_eq!(max_index, 3);
    }
}
