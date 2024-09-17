use std::env::current_dir;
use symbolic_manipulation_engine::build_cli;

use symbolic_manipulation_engine::cli::cli::{Cli, CliMode};
use symbolic_manipulation_engine::cli::filesystem::FileSystem;

fn main() {
    env_logger::init();

    let matches = build_cli().get_matches();

    let current_directory = match current_dir() {
        Ok(path) => path,
        Err(e) => {
            println!("Couldn't get current directory: {}", e);
            return;
        }
    };

    let filesystem = FileSystem::new(current_directory);
    let mut cli = Cli::new(filesystem, CliMode::Production);

    let result = match matches.subcommand() {
        Some(("init", _sub_matches)) => cli.init(),
        Some(("rmws", sub_matches)) => cli.rmws(sub_matches),
        Some(("ls", _sub_matches)) => cli.ls(),
        Some(("import-context", sub_matches)) => cli.import_context(sub_matches),
        Some(("export-context", sub_matches)) => cli.export_context(sub_matches),
        Some(("ls-contexts", _sub_matches)) => cli.ls_contexts(),
        Some(("add-interpretation", sub_matches)) => cli.add_interpretation(sub_matches),
        Some(("remove-interpretation", sub_matches)) => cli.remove_interpretation(sub_matches),
        Some(("add-type", sub_matches)) => cli.add_type(sub_matches),
        Some(("add-algorithm", sub_matches)) => cli.add_algorithm(sub_matches),
        Some(("add-transformation", sub_matches)) => cli.add_transformation(sub_matches),
        Some(("add-joint-transformation", sub_matches)) => {
            cli.add_joint_transformation(sub_matches)
        }
        Some(("get-transformations", sub_matches)) => cli.get_transformations(sub_matches),
        Some(("get-transformations-from", sub_matches)) => {
            cli.get_transformations_from(sub_matches)
        }
        Some(("hypothesize", sub_matches)) => cli.hypothesize(sub_matches),
        Some(("derive", sub_matches)) => cli.derive(sub_matches),
        Some(("undo", _sub_matches)) => cli.undo(),
        Some(("redo", _sub_matches)) => cli.redo(),
        _ => Err("No subcommand was provided".to_string()),
    };

    match result {
        Ok(message) => println!("{}", message),
        Err(e) => eprintln!("{}", e),
    }
}
