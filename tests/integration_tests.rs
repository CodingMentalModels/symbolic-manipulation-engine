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
        build_cli().get_matches_from(vec!["symbolic-manipulation-engine", "derive", "--", "x=4"]);
    cli.derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
    cli.ls().unwrap();
}

#[test]
fn test_algorithm_applies() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new("tests\\assets\\test_algorithm_applies\\"));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "x+y=15",
    ]);
    cli.derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
    cli.ls().unwrap();
}

#[test]
fn test_adds_to_both_sides() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new("tests\\assets\\test_adds_to_both_sides\\"));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "x+y=x+10",
    ]);
    cli.derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
    cli.ls().unwrap();
}

#[test]
fn test_applies_joint_transforms() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new("tests\\assets\\test_applies_joint_transforms\\"));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "x+y=5+10",
    ]);
    cli.derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
    cli.ls().unwrap();
}

#[test]
fn test_duplicates_interpretations() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "tests\\assets\\test_duplicates_interpretations\\",
    ));
    let filesystem = FileSystem::new(dir);
    let mut cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "add-type",
        "--",
        "Integer",
    ]);
    cli.add_type(matches.subcommand_matches("add-type").unwrap())
        .unwrap();
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "add-interpretation",
        "--",
        "Singleton",
        "1",
        "Integer",
        "a",
    ]);
    cli.add_interpretation(matches.subcommand_matches("add-interpretation").unwrap())
        .unwrap();
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "duplicate-interpretation",
        "--",
        "0",
        "b",
    ]);
    cli.duplicate_interpretation(
        matches
            .subcommand_matches("duplicate-interpretation")
            .unwrap(),
    )
    .unwrap();
    let workspace_store = cli.load_workspace_store().unwrap();
    let workspace = workspace_store.compile();
    assert_eq!(workspace.get_interpretations().len(), 2);
}
