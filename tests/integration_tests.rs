use std::{env::current_dir, path::Path};

use symbolic_manipulation_engine::{
    self, build_cli,
    cli::{
        cli::{Cli, CliMode},
        filesystem::FileSystem,
    },
};

#[test]
fn test_add_algorithm() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new("tests\\assets\\test_add_algorithm_works\\"));
    let filesystem = FileSystem::new(dir);
    let mut cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "add-algorithm",
        "--",
        "Addition",
        "+",
        "Real",
    ]);
    cli.add_algorithm(matches.subcommand_matches("add-algorithm").unwrap())
        .unwrap();

    let matches =
        build_cli().get_matches_from(vec!["symbolic-manipulation-engine", "derive", "--", "4"]);
    cli.derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
}
