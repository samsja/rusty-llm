# rusty llm

Run LLM locally on cpu. Written in rust.

## Usage


To use, first you need to build the binary.
```bash
cargo build --release
```

Then you can run via
```bash
cargo run 10 
```

the `10` is a parameter that control the number of tokens that will be produced.

```bash
========== GPT 2 ================
Please enter your prompt: 

What is the capital of France?
```

GPT2 will generate the following text:
```bash
========== GPT 2 ================
Please enter your prompt: 

What is the capital of France?

The capital of France is Paris.
```
