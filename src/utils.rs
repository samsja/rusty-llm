use ndarray::{Array, Ix2, NdFloat};

pub fn tril<'a, T: NdFloat>(x: &'a mut Array<T, Ix2>) -> &'a Array<T, Ix2> {
    // similar to numpy or torch tril
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            if j > i {
                x[[i, j]] = T::from(0.0).unwrap();
            }
        }
    }

    x
}

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
}
