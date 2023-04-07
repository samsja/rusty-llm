use crate::convert::{from_safe_tensorview, from_safe_tensorview_1d};
use crate::float::MyFloat;
use crate::nn::block::Block;
use crate::nn::head::CausalHead;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::linear::{Linear, LinearNoBias};
use crate::nn::utils::{argmax, softmax};
use ndarray::{s, Array, Axis, Ix2};
use safetensors::SafeTensors;

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

    pub fn generate(&self, indices: &Vec<usize>) -> usize {
        let embed = self.forward(indices);
        let logits = embed.slice(s![-1, ..]); // only keep the last work embedding
        let probs = softmax(&logits);
        argmax(&probs.view())
    }

    pub fn load_linear(tensors: &SafeTensors, weight_name: &str, bias_name: &str) -> Linear<T> {
        let weight = from_safe_tensorview::<T>(tensors.tensor(weight_name).unwrap());

        let bias = from_safe_tensorview_1d::<T>(tensors.tensor(bias_name).unwrap());

        Linear::<T>::new(weight, bias)
    }

    pub fn load_layer_norm(
        tensors: &SafeTensors,
        weight_name: &str,
        bias_name: &str,
    ) -> LayerNorm<T> {
        let weight = from_safe_tensorview_1d::<T>(tensors.tensor(weight_name).unwrap());

        let bias = from_safe_tensorview_1d::<T>(tensors.tensor(bias_name).unwrap());

        LayerNorm::<T>::new(weight, bias)
    }

    pub fn load_block(tensors: &SafeTensors, index: usize) -> Block<T> {
        let qkv = GPT::<T>::load_linear(
            tensors,
            &format!("h.{}.attn.c_attn.weight", index),
            &format!("h.{}.attn.c_attn.bias", index),
        );

        let proj_head = GPT::<T>::load_linear(
            tensors,
            &format!("h.{}.attn.c_proj.weight", index),
            &format!("h.{}.attn.c_proj.bias", index),
        );

        let head = CausalHead::<T>::new(qkv, proj_head);

        let fc = GPT::<T>::load_linear(
            tensors,
            &format!("h.{}.mlp.c_fc.weight", index),
            &format!("h.{}.mlp.c_fc.bias", index),
        );

        let proj = GPT::<T>::load_linear(
            tensors,
            &format!("h.{}.mlp.c_proj.weight", index),
            &format!("h.{}.mlp.c_proj.bias", index),
        );

        let ln_1 = GPT::<T>::load_layer_norm(
            tensors,
            &format!("h.{}.ln_1.weight", index),
            &format!("h.{}.ln_1.bias", index),
        );

        let ln_2 = GPT::<T>::load_layer_norm(
            tensors,
            &format!("h.{}.ln_2.weight", index),
            &format!("h.{}.ln_2.bias", index),
        );

        Block::<T>::new(ln_1, head, ln_2, fc, proj)
    }

    pub fn load_from_safe_tensors(tensors: &SafeTensors, num_block: usize) -> GPT<T> {
        let w_token_embed = from_safe_tensorview::<T>(tensors.tensor("wte.weight").unwrap());

        let w_pos_embed = from_safe_tensorview::<T>(tensors.tensor("wpe.weight").unwrap());

        // use 11
        let blocks = (0..num_block)
            .map(|i| GPT::<T>::load_block(tensors, i))
            .collect::<Vec<Block<T>>>();

        let next_word_weight = from_safe_tensorview::<T>(tensors.tensor("wte.weight").unwrap());
        let next_word_layer = LinearNoBias::<T>::new(next_word_weight);

        GPT::<T>::new(w_token_embed, w_pos_embed, blocks, next_word_layer)
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

        gpt.generate(&ids);
    }

    use std::fs::File;
    use std::io::prelude::*;

    #[test]
    fn test_weight_loading() {
        let mut f = File::open("models/model.safetensors").unwrap();
        let mut buffer = Vec::new();

        // read the whole file
        f.read_to_end(&mut buffer).unwrap();
        let tensors: SafeTensors = SafeTensors::deserialize(&buffer).unwrap();

        let gpt = GPT::<f32>::load_from_safe_tensors(&tensors, 2);

        let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").unwrap();

        let encode = tokenizer.encode("hello world", false).unwrap();

        let mut ids: Vec<usize> = encode.get_ids().iter().map(|&x| x as usize).collect();

        ids.push(gpt.generate(&ids));
    }
}
