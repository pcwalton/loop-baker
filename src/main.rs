// loop-baker/src/main.rs

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    input: PathBuf,
}

fn main() {
    let args = Args::parse();
    println!("Hello, world!");
}
