fn main() -> anyhow::Result<()> {
    reglyco_cli::run_with_version(env!("CARGO_PKG_VERSION"))
}
