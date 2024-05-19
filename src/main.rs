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
        .subcommand(Command::new("add-interpretation").about("Takes a condition (string to match), expression type, precedence, and output type and adds it as an interpretation.")
            .arg(
                Arg::new("condition").required(true).help("String which the parser will match to trigger the interpretation.")
            ).arg(
                Arg::new("expression-type").required(true).help("Singleton, Prefix, Infix, Postfix, or Functional (case insensitive)")
            ).arg(
                Arg::new("precedence").required(true).help("Relative precedence for different interpretations. Higher values have higher precedence.")
            ).arg(
                Arg::new("output-type").required(true).help("Output type to parse into.")
            ))
        .subcommand(Command::new("hypothesize").about("Takes a statement and adds it to the Workspace as a new hypothesis").arg(
                Arg::new("statement").required(true)
                )
                    )
        .subcommand(Command::new("get-transformations").about("Takes a partial statement and gets all valid transformations sorted based on the string.").arg(
                Arg::new("partial-statement").required(true)
                )
                    )
        .subcommand(Command::new("derive").about("Checks if the provided statement is valid and adds it to the Workspace if so.").arg(
                Arg::new("statement").required(true)
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
    let mut cli = Cli::new(filesystem);

    let result = match matches.subcommand() {
        Some(("init", _sub_matches)) => cli.init(),
        Some(("rmws", sub_matches)) => cli.rmws(sub_matches),
        Some(("ls", _sub_matches)) => cli.ls(),
        Some(("add-interpretation", sub_matches)) => cli.add_interpretation(sub_matches),
        Some(("get-transformations", sub_matches)) => cli.get_transformations(sub_matches),
        Some(("derive", sub_matches)) => cli.derive(sub_matches),
        _ => Err("No subcommand was provided".to_string()),
    };

    match result {
        Ok(message) => println!("{}", message),
        Err(e) => eprintln!("{}", e),
    }
}
