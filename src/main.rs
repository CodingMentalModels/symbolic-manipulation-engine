mod symbol;
mod workspace;
mod cli;
mod config;

use std::{env::current_dir};
use clap::{arg, Command, Arg, Subcommand, ArgAction};
use config::{STATE_DIRECTORY_RELATIVE_PATH};
use cli::cli::Cli;

use crate::workspace::{workspace::Workspace};
use crate::cli::{filesystem::FileSystem};

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
        ).subcommand(
            Command::new("rmws")
                .about("Removes the workspace")
                .arg(
                    Arg::new("force")
                        .short('f')
                        .long("force")
                        .action(ArgAction::SetTrue)
                        .help("Forces the removal of the workspace")
                )
        ).get_matches();

        let current_directory = match current_dir() {
            Ok(path) => path,
            Err(e) => {
                println!("Couldn't get current directory: {}", e);
                return;
            },
        };

        let filesystem = FileSystem::new(current_directory);
        let cli = Cli::new(filesystem);

        let result = match matches.subcommand() {
            Some(("init", _sub_matches)) => {
                cli.init()
            },
            Some(("rmws", sub_matches)) => {
                cli.rmws(sub_matches)
            },
            _ => {
                Err("No subcommand was used".to_string())
            }
        };

        match result {
            Ok(message) => println!("{}", message),
            Err(e) => println!("{}", e),
        }


}
