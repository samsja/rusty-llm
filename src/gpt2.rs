use crate::float::MyFloat;
use crate::nn::block::Block;
use crate::nn::linear::LinearNoBias;
use ndarray::{Array, Axis, Ix2};

pub struct GPT<T>
where
    T: MyFloat,
{
    w_token_embed: Array<T, Ix2>,
    w_pos_embed: Array<T, Ix2>,
    blocks: Vec<Block<T>>,
    next_word_layer: LinearNoBias<T>, // TODO: should be without bias
}

impl<T> GPT<T>
where
    T: MyFloat,
{
    pub fn new(
        w_token_embed: Array<T, Ix2>,
        w_pos_embed: Array<T, Ix2>,
        blocks: Vec<Block<T>>,
        next_word_layer: LinearNoBias<T>,
    ) -> GPT<T> {
        GPT::<T> {
            w_token_embed,
            w_pos_embed,
            blocks,
            next_word_layer,
        }
    }

    pub fn forward(&self, indices: &Vec<usize>) -> Array<T, Ix2> {
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

    use tokenizers::Tokenizer;

    #[test]
    fn test_gpt() {
        let embed_dim = 16;
        //  let seq_len = 10;
        let vocab_size = 50257;
        // let num_head = 4;
        let block_size = 1024;
        let n_blocks = 3;

        let w_token_embed = Array::<f32, _>::zeros((vocab_size, embed_dim).f());
        let w_pos_embed = Array::<f32, _>::zeros((block_size, embed_dim).f());

        let blocks = (0..n_blocks)
            .map(|_| Block::<f32>::new_zeros(embed_dim))
            .collect::<Vec<Block<f32>>>();

        let next_word_layer = LinearNoBias::<f32>::new_zeros(embed_dim, vocab_size);

        let gpt = GPT::<f32>::new(w_token_embed, w_pos_embed, blocks, next_word_layer);

        let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").unwrap();

        let encode = tokenizer.encode("hello", false).unwrap();

        let ids: Vec<usize> = encode.get_ids().iter().map(|&x| x as usize).collect();

        gpt.forward(&ids);
    }

    use safetensors::tensor::TensorView;
    use safetensors::SafeTensors;

    use std::fs::File;
    use std::io::prelude::*;

    pub fn to_f32<'data>(view: TensorView<'data>) -> Vec<f32> {
        // inspire from smelte-rs.
        // https://github.com/Narsil/smelte-rs/blob/d81f714abce2e64539d3c87dfc6c5488e6a65c03/examples/bert.rs#L168
        // TODO understand what the fuck is going on here. I want to move on towards the end of
        // the project and then deal with this part

        let v = view.data();

        let mut c = Vec::with_capacity(v.len() / 4);
        let mut i = 0;
        while i < v.len() {
            c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
            i += 4;
        }
        c
    }

    #[test]
    fn test_weight_loading() {
        let mut f = File::open("models/model.safetensors").unwrap();
        let mut buffer = Vec::new();

        // read the whole file
        f.read_to_end(&mut buffer).unwrap();
        let tensors: SafeTensors = SafeTensors::deserialize(&buffer).unwrap();
        let t = Array::<f32, _>::from(to_f32(tensors.tensor(&format!("h.0.attn.bias")).unwrap()));
        let t = Array::<f32, _>::from(to_f32(
            tensors.tensor(&format!("h.0.attn.c_attn.weight")).unwrap(),
        ));
    }
}
