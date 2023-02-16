mod symbol;
mod workspace;

use std::{env::current_dir};
use clap::{arg, Command, Arg, Subcommand};


fn main() {
    let matches = Command::new("Symbolic Manipulation Engine")
        .version("0.0")
        .author("Coding Mental Models <codingmentalmodels@gmail.com>")
        .about("Symbolic Manipulation Engine for performing mathematical operations")
        .propagate_version(true)
        .subcommand_required(true)
        .subcommand(
            Command::new("init")
                .about("Initializes a new workspace")
        ).get_matches();

        match matches.subcommand() {
            Some(("init", sub_matches)) => {
                println!("Initializing new workspace in {}", current_dir().unwrap().display());
            },
            _ => {
                println!("No subcommand was used");
            }
        }


}
