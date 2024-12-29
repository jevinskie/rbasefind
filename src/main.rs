extern crate byteorder;
extern crate fnv;
extern crate rbasefind;
extern crate regex;

use clap::Parser;
use rbasefind::Config;
use std::process;

fn main() {
    let config = Config::parse();

    if let Err(e) = rbasefind::run(config) {
        eprintln!("Application error: {}", e);
        process::exit(1);
    }
}
