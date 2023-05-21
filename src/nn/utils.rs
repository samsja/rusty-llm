use crate::float::MyFloat;
use ndarray::{s, Array, ArrayView, ArrayViewMut, Ix1, Ix2, Ix3, IxDyn};

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

pub fn fill_tril_3d<'a, T: MyFloat>(x: &'a mut Array<T, Ix3>, val: T) -> &'a mut Array<T, Ix3> {
    // similar to numpy or torch tril
    for k in 0..x.shape()[0] {
        for i in 0..x.shape()[1] {
            for j in 0..x.shape()[2] {
                if j > i {
                    x[[k, i, j]] = val;
                }
            }
        }
    }

    x
}

pub fn tril<'a, T: MyFloat>(x: &'a mut Array<T, Ix2>) -> &'a Array<T, Ix2> {
    fill_tril(x, T::from(0.0).unwrap())
}

pub fn softmax<T: MyFloat>(x: &ArrayView<T, Ix1>) -> Array<T, Ix1> {
    let max_ = max_val(&x.into_dyn());

    let exp_x = x.mapv(|x| (x - max_).exp());

    let sum = exp_x.sum();
    exp_x / sum
}

pub fn softmax_inplace<T: MyFloat>(x: &mut ArrayViewMut<T, Ix1>) {
    let max_ = max_val(&x.view().into_dyn());

    x.mapv_inplace(|a| (a - max_).exp());

    let sum = x.sum();
    x.mapv_inplace(|a| a / sum);
}

pub fn softmax_inplace_3d<T: MyFloat>(x: &mut Array<T, Ix3>) {
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            let mut base = x.slice_mut(s![i, j, ..]);
            softmax_inplace(&mut base);
        }
    }
}

pub fn max<T: MyFloat>(x: &ArrayView<T, IxDyn>) -> (usize, T) {
    let (index, max_val) = x
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (index, *max_val)
}

pub fn argmax<T: MyFloat>(x: &ArrayView<T, IxDyn>) -> usize {
    let (index, _) = max(x);
    index
}

pub fn max_val<T: MyFloat>(x: &ArrayView<T, IxDyn>) -> T {
    let (_, val) = max(x);
    val
}

#[macro_export]
macro_rules! time_it {
    ($context:literal, $s:stmt) => {
        let timer = std::time::Instant::now();
        $s
        println!("{}: {:?}", $context, timer.elapsed());
    };
}
// credits to https://notes.iveselov.info/programming/time_it-a-case-study-in-rust-macros

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
    fn test_trill_3d() {
        let mut mat1 = Array::<f32, Ix3>::ones((2, 3, 3).f());
        fill_tril_3d(&mut mat1, 0.0);

        assert_eq!(
            mat1,
            Array::<f32, Ix3>::from(vec![
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
            ])
        );
    }

    #[test]
    fn test_argmax() {
        let index = Array::<f32, _>::from(vec![1.0, 2.0, 3.0, 12.0, 1.0]);
        let max_index = argmax(&index.view().into_dyn());
        assert_eq!(max_index, 3);
    }

    #[test]
    fn test_softmax_inplace() {
        let mut mat1 = Array::<f32, _>::from(vec![1.0, -1e9, -1e9]);
        softmax_inplace(&mut mat1.view_mut());
        assert_eq!(mat1, Array::<f32, _>::from(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_softmax_1d() {
        let mat1 = Array::<f32, _>::from(vec![1.0, -1e9, -1e9]);
        let mat2 = softmax(&mat1.view());
        assert_eq!(mat2, Array::<f32, _>::from(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_softmax_3d() {
        let mut mat1 = Array::<f32, Ix3>::ones((2, 3, 3).f());
        fill_tril_3d(&mut mat1, -1e9);

        softmax_inplace_3d(&mut mat1);

        assert_eq!(
            mat1,
            Array::<f32, _>::from(vec![
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.33333334, 0.33333334, 0.33333334]
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.33333334, 0.33333334, 0.33333334]
                ]
            ])
        );
    }

    #[test]
    fn test_time_it() {
        time_it!("println", println!("hello, world!"));

        let x = 1;
        time_it!("let", let y = x + 2);
        let _z = y;
    }
}
