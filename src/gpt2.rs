use crate::float::MyFloat;
use crate::nn::block::Block;
use crate::nn::linear::Linear;
use ndarray::{Array, Axis, Ix2};

pub struct GPT<T>
where
    T: MyFloat,
{
    w_token_embed: Array<T, Ix2>,
    w_pos_embed: Array<T, Ix2>,
    blocks: Vec<Block<T>>,
    next_word_layer: Linear<T>,
}

impl<T> GPT<T>
where
    T: MyFloat,
{
    pub fn new(
        w_token_embed: Array<T, Ix2>,
        w_pos_embed: Array<T, Ix2>,
        blocks: Vec<Block<T>>,
        next_word_layer: Linear<T>,
    ) -> GPT<T> {
        GPT::<T> {
            w_token_embed,
            w_pos_embed,
            blocks,
            next_word_layer,
        }
    }

    pub fn forward(&self, indices: &[usize]) -> Array<T, Ix2> {
        let token_embeddng = self.w_token_embed.select(Axis(0), indices);
        let range: Vec<usize> = (0..indices.len()).map(|i| i).collect();
        let pos_embedding = self.w_pos_embed.select(Axis(0), &range);
        let embed = pos_embedding + token_embeddng; // TODO : optimization do addition in place

        let mut output = embed;
        for block in self.blocks.iter() {
            output = block.forward(&output);
        }

        self.next_word_layer.forward(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_gpt() {
        let embed_dim = 16;
        //  let seq_len = 10;
        let vocab_size = 32;
        // let num_head = 4;
        let block_size = 8;
        let n_blocks = 3;

        let w_token_embed = Array::<f32, _>::zeros((vocab_size, embed_dim).f());
        let w_pos_embed = Array::<f32, _>::zeros((block_size, embed_dim).f());

        let blocks = (0..n_blocks)
            .map(|_| Block::<f32>::new_zeros(embed_dim))
            .collect::<Vec<Block<f32>>>();

        let next_word_layer = Linear::<f32>::new_zeros(embed_dim, vocab_size);

        let gpt = GPT::<f32>::new(w_token_embed, w_pos_embed, blocks, next_word_layer);

        gpt.forward(&[1, 3, 4]);
    }
}
