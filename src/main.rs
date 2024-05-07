mod cli;
mod config;
mod constants;
mod context;
mod parsing;
mod symbol;
mod workspace;

use clap::{arg, Arg, ArgAction, Command, Subcommand};
use cli::cli::Cli;
use config::STATE_DIRECTORY_RELATIVE_PATH;
use std::env::current_dir;

use crate::cli::filesystem::FileSystem;
use crate::workspace::workspace::Workspace;

fn main() {
    let matches = Command::new("Symbolic Manipulation Engine")
        .version("0.0")
        .author("Coding Mental Models <codingmentalmodels@gmail.com>")
        .about("Symbolic Manipulation Engine for performing mathematical operations")
        .propagate_version(true)
        .subcommand_required(true)
        .subcommand(
            Command::new("hello-world").about("Returns Hello World to stdout for debugging."),
        )
        .subcommand(Command::new("init").about("Initializes a new workspace"))
        .subcommand(
            Command::new("rmws").about("Removes the workspace").arg(
                Arg::new("force")
                    .short('f')
                    .long("force")
                    .action(ArgAction::SetTrue)
                    .help("Forces the removal of the workspace"),
            ),
        )
        .subcommand(Command::new("ls").about("Lists the workspaces"))
        .subcommand(Command::new("get-transformations").about("Takes a partial string and gets all valid transformations sorted based on the string.").arg(
                Arg::new("string").required(true)
                )
                    )
        .get_matches();

    let current_directory = match current_dir() {
        Ok(path) => path,
        Err(e) => {
            println!("Couldn't get current directory: {}", e);
            return;
        }
    };

    let filesystem = FileSystem::new(current_directory);
    let cli = Cli::new(filesystem);

    let result = match matches.subcommand() {
        Some(("hello-world", _sub_matches)) => cli.hello_world(),
        Some(("init", _sub_matches)) => cli.init(),
        Some(("rmws", sub_matches)) => cli.rmws(sub_matches),
        Some(("ls", _sub_matches)) => cli.ls(),
        Some(("get-transformations", sub_matches)) => cli.get_transformations(sub_matches),
        _ => Err("No subcommand was used".to_string()),
    };

    match result {
        Ok(message) => println!("{}", message),
        Err(e) => eprintln!("{}", e),
    }
}
