use ndarray::{Array, Ix2, Ix3, NdFloat};

pub fn dot_3d<'a, T: NdFloat>(mat1: &'a Array<T, Ix3>, mat2: &'a Array<T, Ix2>) -> Array<T, Ix3> {
    let shape = mat1.shape();

    let mat1_2d = mat1.to_shape((shape[0] * shape[1], shape[2])).unwrap();

    let dot_product = mat1_2d.dot(mat2);

    let shape_product = dot_product.shape();
    let new_shape = (shape[0], shape[1], shape_product[1]);

    let a = dot_product.into_shape(new_shape).unwrap();

    a.into_dimensionality::<Ix3>().unwrap() // probably there is a way to not have to do this
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_dot_3d() {
        let mat1 = Array::<f32, _>::zeros((3, 10, 12).f());
        let mat2 = Array::<f32, _>::zeros((12, 15).f());
        let mat3 = dot_3d(&mat1, &mat2);

        assert_eq!(mat3.shape(), &[3, 10, 15]);
    }
}
