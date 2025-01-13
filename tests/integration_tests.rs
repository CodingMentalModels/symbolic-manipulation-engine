use std::{env::current_dir, path::Path};

use symbolic_manipulation_engine::{
    self, build_cli,
    cli::{
        cli::{Cli, CliMode},
        filesystem::FileSystem,
    },
};

#[test]
fn test_gets_valid_transformations_from_scoped_expression() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "tests\\assets\\test_gets_valid_transformations_from_scoped_expression\\",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "get-transformations-from",
        "--",
        "0",
        "[0]",
        "[4,6,7]",
    ]);
    let result = cli
        .get_transformations_from(
            matches
                .subcommand_matches("get-transformations-from")
                .unwrap(),
        )
        .unwrap();
    assert_eq!(result, "[\"((y*x)=z)\"]");
}

#[test]
fn test_gets_valid_transformations_large_expression_scoped() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "tests\\assets\\test_gets_valid_transformations_large_expression_scoped\\",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "get-transformations",
        "--",
        "a*x_0^2+b*x_0+c=a*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))^2+b*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))+c",
        "[0,1]",
        "[19]",
    ]);
    let result = cli
        .get_transformations(matches.subcommand_matches("get-transformations").unwrap())
        .unwrap();
    assert_eq!(result, "[\"((((a*(x_0^2))+(b*x_0))+c)=(((a*(((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))^2))+(b*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))))+c))\"]");
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "((((a*(x_0^2))+(b*x_0))+c)=(((a*(((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))^2))+(b*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))))+c))",
        "[0,1]",
        "[19]",
    ]);
    let result = cli
        .derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
}

#[test]
fn test_substitutes_large_expression() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "tests\\assets\\test_substitutes_large_expression\\",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "a*(x_0)^2+b*(x_0)+c=a*(x_0)^2+b*((Negative(b)+(b^2-4*a*c))/(2*a))+c",
    ]);
    let _result = cli
        .derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
}

#[test]
fn test_substitutes_using_substitution_axiom() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "tests\\assets\\test_substitutes_using_substitution_axiom\\",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "15+y=7",
    ]);
    let _result = cli
        .derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
}

#[test]
fn test_substitutes_in_any_order() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new("tests\\assets\\test_substitutes_in_any_order\\"));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "derive",
        "--",
        "x+y=y+x",
        "[1]",
        "[0]",
    ]);
    let _result = cli
        .derive(matches.subcommand_matches("derive").unwrap())
        .unwrap();
}
#[test]
fn test_evaluation_works() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new("tests\\assets\\test_evaluation_works\\"));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "evaluate",
        "--",
        "2+(3-7*5)", // = 2+(3-35) = 5 - 35 = -30
    ]);
    let result = cli
        .evaluate(matches.subcommand_matches("evaluate").unwrap())
        .unwrap();
    assert_eq!(result, "-30".to_string());
}

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
        "a",
        "Integer",
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

#[test]
fn test_gets_valid_transformations_large_expression_scoped_2() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "C:\\Users\\cmsdu\\repos\\symbolic-manipulation-engine\\tests\\assets\\test_gets_valid_transformations_large_expression_scoped_2",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "get-transformations",
        "--",
        "a*x_0^2+b*x_0+c=a*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))^2+b*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))+c",
        "[0]",
        "[19]",
    ]);
    let result = cli
        .get_transformations(matches.subcommand_matches("get-transformations").unwrap())
        .unwrap();
    assert_eq!(result, "[\"((((a*(x_0^2))+(b*x_0))+c)=(((a*(((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))^2))+(b*((Negative(b)+(((b^2)-((4*a)*c))^(1/2)))/(2*a))))+c))\"]");
}

#[test]
fn test_substitutes_with_multiple_variables() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "C:\\Users\\cmsdu\\repos\\symbolic-manipulation-engine\\tests\\assets\\test_substitutes_with_multiple_variables",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "get-transformations",
        "--",
        "((((g+r)+y)+b)=((((6*r)+r)+y)+b))",
    ]);
    let result = cli
        .get_transformations(matches.subcommand_matches("get-transformations").unwrap())
        .unwrap();
    assert_eq!(result, "[\"((((g+r)+y)+b)=((((6*r)+r)+y)+b))\"]");
}

#[test]
fn test_removes_statements_and_dependents() {
    let root_dir = current_dir().unwrap();
    let dir = root_dir.join(Path::new(
        "C:\\Users\\cmsdu\\repos\\symbolic-manipulation-engine\\tests\\assets\\test_removes_statements_and_dependents",
    ));
    let filesystem = FileSystem::new(dir);
    let cli = Cli::new(filesystem, CliMode::Testing);
    let matches = build_cli().get_matches_from(vec![
        "symbolic-manipulation-engine",
        "remove-statement",
        "--",
        "0",
    ]);
    let result = cli
        .remove_statement(matches.subcommand_matches("remove-statement").unwrap())
        .unwrap();
    assert_eq!(
        result,
        "[\"((x*z)+(y*z))\",\"((x+y)*z)\",\"((z*x)+(z*y))\",\"(z*(x+y))\"]"
    );
}
