use std::fs::File;
use std::io;
use std::io::prelude::*;

use rusty_llm::gpt2::GPT;

use safetensors::SafeTensors;
use tokenizers::Tokenizer;

fn main() {
    let mut f = File::open("models/model.safetensors").unwrap();
    let mut buffer = Vec::new();

    // read the whole file
    f.read_to_end(&mut buffer).unwrap();
    let tensors: SafeTensors = SafeTensors::deserialize(&buffer).unwrap();

    let gpt = GPT::<f32>::load_from_safe_tensors(&tensors, 12);

    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").unwrap();

    let init_text = "hello world";

    let encode = tokenizer.encode(init_text, false).unwrap();

    let mut ids: Vec<usize> = encode.get_ids().iter().map(|&x| x as usize).collect();

    print!("{}", init_text);

    for _ in 0..10 {
        let new_word_id = gpt.generate(&ids);

        ids.push(new_word_id);

        let id_to_decode: Vec<u32> = vec![new_word_id as u32];

        let txt = tokenizer.decode(id_to_decode, false).unwrap();
        print!("{}", txt);
        io::stdout().flush().unwrap(); // flush to see output in real time
    }
}
