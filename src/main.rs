use anyhow::{Context, Result};
use pdf_extract::extract_text;
use rig::client::{CompletionClient, EmbeddingsClient};
use rig::embeddings::EmbeddingsBuilder;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::{client::ProviderClient, providers::openai};
use std::path::Path;
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn load_pdf_content<P: AsRef<Path>>(file_path: P) -> Result<String> {
    extract_text(file_path.as_ref())
        .with_context(|| format!("Failed to extract text from PDF: {:?}", file_path.as_ref()))
}

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);

        if end >= words.len() {
            break;
        }

        start += chunk_size - overlap;
    }

    chunks
}

use clap::Parser;

#[derive(Parser)]
#[command(name = "rag-my-pdf")]
#[command(version, about = "PDF RAG chatbot using OpenAI", long_about = None)]
struct Cli {
    /// Path to the PDF file to load
    #[arg(short, long)]
    pdf: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// OpenAI model to use
    #[arg(short, long, default_value = "gpt-3.5-turbo")]
    model: String,

    /// Chunk size in words
    #[arg(long, default_value = "500")]
    chunk_size: usize,

    /// Overlap between chunks in words
    #[arg(long, default_value = "50")]
    chunk_overlap: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing/logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false).pretty())
        .with(EnvFilter::new(log_level))
        .init();

    info!("Starting RAG PDF Chatbot");
    debug!("Using model: {}", cli.model);

    // This requires the `OPENAI_API_KEY` environment variable to be set.
    info!("Initializing OpenAI client");
    let openai_client = openai::Client::from_env();

    // Load document from PDF if provided, otherwise use default
    let document: String = if let Some(pdf_path) = cli.pdf.clone() {
        info!("Loading PDF from: {}", pdf_path);
        load_pdf_content(&pdf_path)?
    } else {
        warn!("No PDF provided, using default document");
        String::from("The answer to life is 42 by the way")
    };

    // Chunk the text
    info!(
        "Chunking text (size: {}, overlap: {})",
        cli.chunk_size, cli.chunk_overlap
    );
    let chunks = chunk_text(&document, cli.chunk_size, cli.chunk_overlap);
    info!("Created {} chunks from document", chunks.len());
    debug!(
        "First chunk preview: {}...",
        chunks.first().map(|c| &c[..c.len().min(100)]).unwrap_or("")
    );

    info!("Creating embedding model");
    let embedding_model = openai_client.embedding_model("text-embedding-ada-002");

    info!("Building embeddings from {} chunks", chunks.len());
    let mut embeddings_builder = EmbeddingsBuilder::new(embedding_model.clone());
    for chunk in chunks.iter() {
        embeddings_builder = embeddings_builder.document(chunk.clone())?;
    }
    let embeddings = embeddings_builder.build().await?;

    debug!("Creating vector store and index");
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    let index = vector_store.index(embedding_model);

    info!("Initializing RAG agent with model: {}", cli.model);
    let rag_agent = openai_client
            .agent(&cli.model)
            .preamble("You are a helpful assistant that answers questions based on the given context from the provided PDF document.")
            .dynamic_context(2, index)
            .build();

    info!("Starting chatbot interface");
    let chatbot = ChatBotBuilder::new().agent(rag_agent).build();

    // Print welcome message
    println!("           Welcome to RAG PDF Chatbot!");
    println!();
    println!("Loaded {} chunks from your document", chunks.len());
    println!("Using model: {}", cli.model);
    if let Some(pdf_path) = cli.pdf {
        println!("Ask me anything about the document {}", pdf_path);
    }
    println!("Type 'exit' or press Ctrl+C to quit\n");

    chatbot.run().await?;

    info!("Chatbot session ended");

    Ok(())
}
