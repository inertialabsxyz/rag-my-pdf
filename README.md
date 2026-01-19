# RAG My PDF

Chat with your PDF documents using Rig

## Setup

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here
```

2. Run with a PDF:
```bash
cargo run -- --pdf document.pdf
```

## Usage

```bash
# Basic usage
cargo run -- --pdf document.pdf

# Verbose output
cargo run -- -v --pdf document.pdf

# Use different model
cargo run -- --pdf document.pdf --model gpt-4

# Custom chunking
cargo run -- --pdf document.pdf --chunk-size 300 --chunk-overlap 50
```

## Options

- `--pdf` - Path to PDF file
- `--verbose` - Show detailed logs
- `--model` - OpenAI model (default: gpt-3.5-turbo)
- `--chunk-size` - Chunk size in words (default: 500)
- `--chunk-overlap` - Overlap in words (default: 50)
