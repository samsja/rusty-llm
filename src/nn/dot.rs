use crate::float::MyFloat;
use ndarray::parallel::prelude::*;
use ndarray::{stack, Array, ArrayView, Axis, Ix2, Ix3};
use rayon::iter::ParallelExtend;

pub fn dot_3d_2d<'a, T: MyFloat>(
    mat1: &'a Array<T, Ix3>,
    mat2: &'a ArrayView<T, Ix2>,
) -> Array<T, Ix3> {
    // mat1:(B, X, Y)
    // mat2: (Y, Z)
    // output: (B, X, Z)

    let shape = mat1.shape();

    let mat1_2d = mat1.to_shape((shape[0] * shape[1], shape[2])).unwrap();

    let dot_product = mat1_2d.dot(mat2);

    let shape_product = dot_product.shape();
    let new_shape = (shape[0], shape[1], shape_product[1]);

    let a = dot_product.into_shape(new_shape).unwrap();

    a.into_dimensionality::<Ix3>().unwrap() // probably there is a way to not have to do this
}

pub fn dot_3d_3d<'a, T: MyFloat>(
    mat1: &'a ArrayView<T, Ix3>,
    mat2: &'a ArrayView<T, Ix3>,
) -> Array<T, Ix3> {
    // mat1:(B, X, Y)
    // mat2: (B, Y, Z)
    // output: (B, X, Z)
    let n = mat1.shape()[0];
    if n != mat2.shape()[0] {
        panic!("mat1 and mat2 should have the same size for the first axis, they don't mat1: {:?} mat2{:?}", mat1.shape(), mat2.shape())
    }

    let mut inner_dot_2d: Vec<Array<T, Ix2>> = Vec::with_capacity(n);

    for (x, y) in mat1.axis_iter(Axis(0)).zip(mat2.axis_iter(Axis(0))) {
        inner_dot_2d.push(x.dot(&y));
    }

    let inner_dot_2d_view = inner_dot_2d
        .iter()
        .map(|x| x.view())
        .collect::<Vec<ArrayView<T, Ix2>>>();

    stack(Axis(0), &inner_dot_2d_view).unwrap()
}

pub fn dot_3d_3d_par<'a, T: MyFloat>(
    mat1: &'a ArrayView<T, Ix3>,
    mat2: &'a ArrayView<T, Ix3>,
) -> Array<T, Ix3> {
    // mat1:(B, X, Y)
    // mat2: (B, Y, Z)
    // output: (B, X, Z)
    let n = mat1.shape()[0];
    if n != mat2.shape()[0] {
        panic!("mat1 and mat2 should have the same size for the first axis, they don't mat1: {:?} mat2{:?}", mat1.shape(), mat2.shape())
    }

    let a_subviews: Vec<_> = mat1.axis_iter(Axis(0)).collect();
    let b_subviews: Vec<_> = mat2.axis_iter(Axis(0)).collect();

    let n = a_subviews.len();

    let mut inner_dot_2d: Vec<Array<T, Ix2>> = Vec::with_capacity(n);

    inner_dot_2d.par_extend(
        a_subviews
            .par_iter()
            .zip(b_subviews.par_iter())
            .map(|(subview_a, subview_b)| subview_a.dot(subview_b)),
    );

    let inner_dot_2d_view = inner_dot_2d
        .iter()
        .map(|x| x.view())
        .collect::<Vec<ArrayView<T, Ix2>>>();

    stack(Axis(0), &inner_dot_2d_view).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_dot_3d() {
        let mat1 = Array::<f32, _>::zeros((3, 10, 12).f());
        let mat2 = Array::<f32, _>::zeros((12, 15).f());
        let mat3 = dot_3d_2d(&mat1, &mat2.view());

        assert_eq!(mat3.shape(), &[3, 10, 15]);
    }

    #[test]
    fn test_dot_3d_3d() {
        let mat1 = array![
            [[0., 1., 2.], [3., 4., 5.]],
            [[6., 7., 8.], [9., 10., 11.]],
            [[12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.]]
        ];

        let mat2 = array![
            [[0., 10.], [20., 30.], [40., 50.]],
            [[60., 70.], [80., 90.], [100., 110.]],
            [[120., 130.], [140., 150.], [160., 170.]],
            [[180., 190.], [200., 210.], [220., 230.]],
        ];

        let output = dot_3d_3d(&mat1.view(), &mat2.view());

        let expected_output = array![
            [[100., 130.], [280., 400.]],
            [[1720., 1930.], [2440., 2740.]],
            [[5500., 5890.], [6760., 7240.]],
            [[11440., 12010.], [13240., 13900.]]
        ];

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dot_3d_3d_par() {
        let mat1 = array![
            [[0., 1., 2.], [3., 4., 5.]],
            [[6., 7., 8.], [9., 10., 11.]],
            [[12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.]]
        ];

        let mat2 = array![
            [[0., 10.], [20., 30.], [40., 50.]],
            [[60., 70.], [80., 90.], [100., 110.]],
            [[120., 130.], [140., 150.], [160., 170.]],
            [[180., 190.], [200., 210.], [220., 230.]],
        ];

        let output = dot_3d_3d_par(&mat1.view(), &mat2.view());

        let expected_output = array![
            [[100., 130.], [280., 400.]],
            [[1720., 1930.], [2440., 2740.]],
            [[5500., 5890.], [6760., 7240.]],
            [[11440., 12010.], [13240., 13900.]]
        ];

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dot_3d_3d_shape() {
        let mat1 = Array::<f32, _>::zeros((3, 10, 12).f());
        let mat2 = Array::<f32, _>::zeros((3, 12, 15).f());
        let mat3 = dot_3d_3d(&mat1.view(), &mat2.view());

        assert_eq!(mat3.shape(), &[3, 10, 15]);
    }
}
