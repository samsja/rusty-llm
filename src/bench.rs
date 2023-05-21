use std::fs::File;
// use std::io;
use std::io::prelude::*;
use std::time::Instant;

use rusty_llm::gpt2::GPT;

use safetensors::SafeTensors;
use tokenizers::Tokenizer;

// use std::env;
// use std::process;

fn main() {
    // Get command line arguments
    // let args: Vec<String> = env::args().collect();
    //
    // // Ensure an argument is provided
    // if args.len() != 2 {
    //     eprintln!("You must provide exactly one argument.");
    //     process::exit(1);
    // }
    //
    // // Try to parse the argument as an integer
    // let number: i32 = match args[1].parse() {
    //     Ok(v) => v,
    //     Err(e) => {
    //         eprintln!("Failed to parse argument as integer: {}", e);
    //         process::exit(1);
    //     }
    // };

    let mut f = File::open("models/model.safetensors").unwrap();
    let mut buffer = Vec::new();

    // read the whole file
    f.read_to_end(&mut buffer).unwrap();
    let tensors: SafeTensors = SafeTensors::deserialize(&buffer).unwrap();

    let gpt = GPT::<f32>::load_from_safe_tensors(&tensors, 12);

    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").unwrap();

    // println!("========== GPT 2 ================");
    //
    // println!("Please enter your prompt: \n");
    //
    // // Make sure the prompt immediately appears on the screen.
    // io::stdout().flush().unwrap();
    //
    // let mut init_text = String::new();
    //
    // io::stdin()
    //     .read_line(&mut init_text)
    //     .expect("Failed to read line");

    // // Remove the trailing newline.
    // let init_text = init_text.trim();

    let init_text = "What is the capital of france ?";
    let number = 10;

    let encode = tokenizer.encode(init_text, false).unwrap();

    let mut ids: Vec<usize> = encode.get_ids().iter().map(|&x| x as usize).collect();

    let start = Instant::now();

    for _ in 0..number {
        let new_word_id = gpt.generate(&ids);

        ids.push(new_word_id);

        // let id_to_decode: Vec<u32> = vec![new_word_id as u32];
        //
        // let txt = tokenizer.decode(id_to_decode, false).unwrap();
        // print!("{}", txt);
        // io::stdout().flush().unwrap(); // flush to see output in real time
    }
    let duration = start.elapsed();
    println!("Time elapsed : {:?}", duration);

    let ids_to_decode: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
    let txt = tokenizer.decode(ids_to_decode, false).unwrap();
    println!("{}", txt);
}

// fn main() {
//     let mat1 = Array::<f32, _>::zeros((10, 100, 100).f());
//     let mat2 = Array::<f32, _>::zeros((10, 100, 100).f());
//
//     for _ in 0..30 {
//         time_it!("dot3d", let mat3 = dot_3d_3d(&mat1.view(), &mat2.view()));
//     }
//
// }

// fn main() {
//     let mat1 = Array::<f32, _>::zeros((10, 100, 100).f());
//     let mat2 = Array::<f32, _>::zeros((10, 100, 100).f());
//
//     for _ in 0..30 {
//         time_it!("dot3d", let _mat3 = dot_3d_3d_par(&mat1.view(), &mat2.view()));
//     }
//
// }
//
// fn main() {
//     let mat1 = Array::<f32, _>::zeros((100, 100).f());
//     let mat2 = Array::<f32, _>::zeros((100, 100).f());
//
//     for _ in 0..100 {
//         time_it!("dot3d", let mat3 = mat1.dot(&mat2));
//     }
// }
