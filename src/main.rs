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
        .subcommand(Command::new("add-interpretation").about("Takes a condition (string to match), expression type, precedence, and output type and adds it as an interpretation."
            ).arg(
                Arg::new("expression-type").required(true).help("Singleton, Prefix, Infix, Postfix, or Functional (case insensitive)")
            ).arg(
                Arg::new("precedence").required(true).help("Relative precedence for different interpretations. Higher values have higher precedence.")
            ).arg(
                Arg::new("condition").help("String which the parser will match to trigger the interpretation.")
            ).arg(
                Arg::new("output-type").help("Output type to parse into.")
            ).arg(
                Arg::new("any-integer").short('n').action(ArgAction::SetFalse)
            ).arg(
                Arg::new("any-numeric").short('f').action(ArgAction::SetFalse)
                )
            )
        .subcommand(Command::new("remove-interpretation").about("Remove an Interpretation from the Workspace by its index.")
            .arg(
                Arg::new("index").required(true).help("Index of interpretation to remove (0-indexed).")
            ))
        .subcommand(Command::new("add-type").about("Takes a type name and optionally a parent type name and adds the type to the Workspace's Type Hierarchy, defaulting to Object if no parent type is provided.")
            .arg(
                Arg::new("type-name").required(true).help("Name of the type to add.")
            ).arg(
                Arg::new("parent-type-name").help("Optional name of the parent type to add the type to. Must already exist in the Type Hierarchy of the Workspace.")
            ))
        .subcommand(Command::new("hypothesize").about("Takes a statement and adds it to the Workspace as a new hypothesis").arg(
                Arg::new("statement").required(true)
                )
                    )
        .subcommand(Command::new("add-transformation").about("Add a Transformation to the Workspace via a 'from' and a 'to' statement, parsed using the Workspace's Interpretations.")
            .arg(
                Arg::new("from").required(true).help("Starting point of the Transformation.")
            ).arg(
                Arg::new("to").required(true).help("Ending point for the Transformation.")
            ))
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
        Some(("remove-interpretation", sub_matches)) => cli.remove_interpretation(sub_matches),
        Some(("add-type", sub_matches)) => cli.add_type(sub_matches),
        Some(("add-transformation", sub_matches)) => cli.add_transformation(sub_matches),
        Some(("get-transformations", sub_matches)) => cli.get_transformations(sub_matches),
        Some(("hypothesize", sub_matches)) => cli.hypothesize(sub_matches),
        Some(("derive", sub_matches)) => cli.derive(sub_matches),
        _ => Err("No subcommand was provided".to_string()),
    };

    match result {
        Ok(message) => println!("{}", message),
        Err(e) => eprintln!("{}", e),
    }
}
