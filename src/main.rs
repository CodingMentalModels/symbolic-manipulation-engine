mod symbol;
mod workspace;

use std::{env::current_dir};
use clap::{arg, Command, Arg, Subcommand};

use crate::workspace::{workspace::Workspace, filesystem::FileSystem};

const STATE_DIRECTORY_RELATIVE_PATH: &str = ".sme";

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
            Some(("init", _sub_matches)) => {
                let current_directory = match current_dir() {
                    Ok(path) => path,
                    Err(e) => {
                        println!("Couldn't get current directory: {}", e);
                        return;
                    },
                };
                let file_system = FileSystem::new(current_directory);
                if file_system.path_exists(STATE_DIRECTORY_RELATIVE_PATH) {
                    println!("A workspace already exists in this directory");
                    return;
                }
                
                match file_system.create_directory(STATE_DIRECTORY_RELATIVE_PATH) {
                    true => println!("Created directory {}", STATE_DIRECTORY_RELATIVE_PATH),
                    false => {
                        println!("Couldn't create directory {}", STATE_DIRECTORY_RELATIVE_PATH);
                        return;
                    }
                }

                let workspace = Workspace::new();
                match file_system.write_file(STATE_DIRECTORY_RELATIVE_PATH, "workspace.toml", workspace.serialize()) {
                    true => println!("Created workspace.toml"),
                    false => {
                        println!("Couldn't create workspace.toml");
                        return;
                    }
                }
                println!("Initialized new workspace in {}", file_system.get_root_directory_path());
            },
            _ => {
                println!("No subcommand was used");
            }
        }


}
