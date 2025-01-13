use std::env::{self, current_dir};
use symbolic_manipulation_engine::build_cli;

use symbolic_manipulation_engine::cli::cli::{Cli, CliMode};
use symbolic_manipulation_engine::cli::filesystem::FileSystem;

fn main() {
    let matches = build_cli().get_matches();
    let command_string: String = env::args().collect::<Vec<String>>().join(" ");

    let current_directory = match current_dir() {
        Ok(path) => path,
        Err(e) => {
            println!("Couldn't get current directory: {}", e);
            return;
        }
    };

    let filesystem = FileSystem::new(current_directory.clone());
    let mut cli = Cli::new(filesystem, CliMode::Production);

    let mut include_in_history = true;
    let result = match matches.subcommand() {
        Some(("init", _sub_matches)) => cli.init(),
        Some(("rmws", sub_matches)) => cli.rmws(sub_matches),
        Some(("ls", _sub_matches)) => {
            include_in_history = false;
            cli.ls()
        }
        Some(("import-context", sub_matches)) => cli.import_context(sub_matches),
        Some(("export-context", sub_matches)) => cli.export_context(sub_matches),
        Some(("ls-contexts", _sub_matches)) => {
            include_in_history = false;
            cli.ls_contexts()
        }
        Some(("add-interpretation", sub_matches)) => cli.add_interpretation(sub_matches),
        Some(("update-interpretation", sub_matches)) => cli.update_interpretation(sub_matches),
        Some(("duplicate-interpretation", sub_matches)) => {
            cli.duplicate_interpretation(sub_matches)
        }
        Some(("remove-interpretation", sub_matches)) => cli.remove_interpretation(sub_matches),
        Some(("add-type", sub_matches)) => cli.add_type(sub_matches),
        Some(("add-algorithm", sub_matches)) => cli.add_algorithm(sub_matches),
        Some(("add-transformation", sub_matches)) => cli.add_transformation(sub_matches),
        Some(("add-joint-transformation", sub_matches)) => {
            cli.add_joint_transformation(sub_matches)
        }
        Some(("get-transformations", sub_matches)) => {
            include_in_history = false;
            cli.get_transformations(sub_matches)
        }
        Some(("get-transformations-from", sub_matches)) => {
            include_in_history = false;
            cli.get_transformations_from(sub_matches)
        }
        Some(("remove-transformation", sub_matches)) => cli.remove_transformation(sub_matches),
        Some(("hypothesize", sub_matches)) => cli.hypothesize(sub_matches),
        Some(("derive", sub_matches)) => cli.derive(sub_matches),
        Some(("derive-theorem", sub_matches)) => cli.derive_theorem(sub_matches),
        Some(("remove-statement", sub_matches)) => cli.remove_statement(sub_matches),
        Some(("undo", _sub_matches)) => cli.undo(),
        Some(("redo", _sub_matches)) => cli.redo(),
        Some(("evaluate", sub_matches)) => {
            include_in_history = false;
            cli.evaluate(sub_matches)
        }
        Some(("command-history", _sub_matches)) => {
            include_in_history = false;
            cli.get_command_history()
        }
        _ => Err("No subcommand was provided".to_string()),
    };

    match result {
        Ok(message) => {
            if include_in_history {
                match cli.update_command_history(command_string) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{}", e);
                        return;
                    }
                }
            }
            println!("{}", message);
        }
        Err(e) => eprintln!("{}", e),
    }
}
