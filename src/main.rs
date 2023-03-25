use ndarray::prelude::*;
mod head;
use head::Head;

fn main() {
    let embed_dim = 100;
    let seq_len = 10;

    let embed = Array::<f32, _>::zeros((seq_len, embed_dim).f());
    println!("{:?}", embed);

    let head = Head::<f32>::new_zeros(embed_dim, 64, 64);

    let att = head.attention(&embed);

    println!("{:?}", att);
}
