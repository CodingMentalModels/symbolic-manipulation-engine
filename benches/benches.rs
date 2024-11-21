use std::{env::current_dir, path::Path};

use criterion::{criterion_group, criterion_main, Criterion};
use symbolic_manipulation_engine::{
    build_cli,
    cli::{
        cli::{Cli, CliMode},
        filesystem::FileSystem,
    },
};

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

// Benchmarks
fn benchmark_test_substitutes_large_expression(c: &mut Criterion) {
    c.bench_function("test_substitutes_large_expression", |b| {
        b.iter(|| test_substitutes_large_expression())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_test_substitutes_large_expression
}
criterion_main!(benches);
