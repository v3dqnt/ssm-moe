use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use ssm_moe::config::MoEConfig;
use ssm_moe::pipeline::MoEPipeline;
use ssm_moe::server;

#[derive(Parser)]
#[command(name = "ssm-moe", about = "SSM Mixture-of-Experts inference engine")]
struct Cli {
    #[arg(short, long, default_value = "default-session")]
    session: String,

    #[arg(short, long)]
    prompt: Option<String>,

    /// Run interactive REPL
    #[arg(short, long)]
    interactive: bool,

    /// Run as an OpenAI-compatible HTTP server instead of one-shot/REPL mode
    /// — this is the mode Vivianne (or any other harness) should point at.
    #[arg(long)]
    serve: bool,

    /// Port for --serve mode
    #[arg(long, default_value_t = 8090)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("ssm_moe=info".parse()?))
        .init();

    let cli = Cli::parse();
    let config = MoEConfig::default();
    let mut pipeline = MoEPipeline::new(config).await?;

    if cli.serve {
        server::serve(pipeline, cli.port).await?;
    } else if cli.interactive {
        repl(&mut pipeline, &cli.session).await?;
    } else if let Some(prompt) = cli.prompt {
        let output = pipeline.run(&cli.session, &prompt)?;
        println!("{output}");
    } else {
        eprintln!("Pass --prompt <text>, --interactive, or --serve");
        std::process::exit(1);
    }

    Ok(())
}

async fn repl(pipeline: &mut MoEPipeline, session_id: &str) -> Result<()> {
    use std::io::{self, BufRead, Write};
    let stdin = io::stdin();
    let stdout = io::stdout();

    println!("SSM MoE — interactive mode. Ctrl+C to exit.\n");

    for line in stdin.lock().lines() {
        let prompt = line?;
        if prompt.trim().is_empty() {
            continue;
        }
        print!("\n> ");
        stdout.lock().flush()?;

        match pipeline.run(session_id, &prompt) {
            Ok(output) => println!("{output}\n"),
            Err(e) => eprintln!("Error: {e}\n"),
        }
    }

    Ok(())
}
