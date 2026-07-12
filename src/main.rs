mod brain {
    pub mod decompose;
    pub mod gate;
    pub mod native_router;
    pub mod router;
}
mod config;
mod experts {
    pub mod model;
    pub mod registry;
}
mod layers {
    pub mod confidence;
    pub mod critic;
    pub mod fusion;
    pub mod linear_head;
}
mod memory {
    pub mod context;
}
mod pipeline;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use config::MoEConfig;
use pipeline::MoEPipeline;

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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("ssm_moe=info".parse()?))
        .init();

    let cli = Cli::parse();
    let config = MoEConfig::default();
    let mut pipeline = MoEPipeline::new(config, &cli.session).await?;

    if cli.interactive {
        repl(&mut pipeline).await?;
    } else if let Some(prompt) = cli.prompt {
        let output = pipeline.run(&prompt)?;
        println!("{output}");
    } else {
        eprintln!("Pass --prompt <text> or --interactive");
        std::process::exit(1);
    }

    Ok(())
}

async fn repl(pipeline: &mut MoEPipeline) -> Result<()> {
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

        match pipeline.run(&prompt) {
            Ok(output) => println!("{output}\n"),
            Err(e) => eprintln!("Error: {e}\n"),
        }
    }

    Ok(())
}
