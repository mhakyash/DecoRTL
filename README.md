# DecoRTL: A Run-time Decoding Framework for RTL Code Generation with LLMs

Official implementation of **DecoRTL**, a run-time decoding framework for RTL (Register Transfer Level) code generation with Large Language Models. This repository provides a modular Python implementation of contrastive decoding with temperature adaptation for generating Verilog hardware description code using vLLM.

**Paper:** [arXiv:2507.02226](https://arxiv.org/abs/2507.02226)

## Features

- **Contrastive Decoding**: Embedding-based token re-ranking for diverse generation
- **Temperature Adaptation**: Dynamic temperature adjustment based on token categories
- **vLLM Integration**: Connects to vLLM server via API

## Installation

```bash
pip install -r requirements.txt
```

## Setup

### 1. Start vLLM Server

On your server machine, start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --port 8000
```

### 2. Setup Tunnel (if needed)

If the server is remote, set up a tunnel to localhost:8000:

```bash
# Example with SSH
ssh -L 8000:localhost:8000 user@server
```

### 3. Load Tokenizer Locally

The tokenizer needs to be loaded locally for tokenization/decoding:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
```

## Usage

### Single Prompt Generation

```python
from example_single import main
main()
```

Or use the API directly:

```python
from transformers import AutoTokenizer
from src.config import DecodingConfig, VLLMConfig, GenerationConfig, TemperatureAdaptationConfig
from src.generator import VerilogGenerator
from src.vllm_client import TokenizerWrapper

# Setup
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
tokenizer_wrapper = TokenizerWrapper(tokenizer)

config = DecodingConfig(
    method="contrastive",
    use_temperature_adaptation=True,
    vllm_config=VLLMConfig(base_url="http://localhost:8000/v1"),
    generation_config=GenerationConfig(base_temperature=0.7, top_k=5),
    temp_adaptation_config=TemperatureAdaptationConfig(enabled=True)
)

generator = VerilogGenerator(
    config=config,
    tokenizer=tokenizer_wrapper,
    embedding_model=None  # See note below
)

# Generate
result = generator.generate("Your prompt here")
print(result["code"])
```

### Batch Processing

```bash
python main.py \
    --input-dir ./prompts \
    --output-dir ./outputs \
    --entropy-dir ./entropy_results \
    --tokenizer-name Qwen/Qwen2.5-Coder-14B-Instruct \
    --method contrastive \
    --use-temp-adaptation \
    --base-temp 0.7 \
    --top-k 5 \
    --resume
```

## Configuration

### Decoding Methods

1. **Contrastive Decoding** (`--method contrastive`)
   - Uses embedding-based re-ranking
   - Requires embedding model (see below)
   - Better diversity and quality

2. **Base Decoding** (`--method base`)
   - Standard temperature-based sampling
   - No embedding model needed
   - Faster but less diverse

### Temperature Adaptation

Temperature adaptation adjusts temperature based on token categories:

- **Structural tokens** (`module`, `endmodule`, `input`, `output`, `wire`, `reg`, `assign`, `always`, `if`, `else`, `case`, `for`, `while`, `begin`, `end`, `posedge`, `negedge`, and punctuation `;`, `,`, `.`, `[`, `]`, `(`, `)`, `{`, `}`, etc.): Lower temperature (more deterministic)
- **High-impact tokens** (`+`, `-`, `*`, `/`, `&`, `|`, `^`, `~`, `!`, `=`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `?`, `:`, `=>`, `&&`, `||`): Higher temperature (more diverse)
- **Other tokens**: Base temperature

Enable with `--use-temp-adaptation` flag.

## Embedding Model for Contrastive Decoding

For contrastive decoding, you need an embedding model. Options:

### Option 1: Use the Same Model's Embeddings

If you have access to the model locally:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
generator = VerilogGenerator(
    config=config,
    tokenizer=tokenizer_wrapper,
    embedding_model=model  # Use model's embedding layer
)
```

### Option 2: Use a Separate Embedding Model

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = VerilogGenerator(
    config=config,
    tokenizer=tokenizer_wrapper,
    embedding_model=embedding_model
)
```

### Option 3: Extend vLLM with Custom Endpoint

Create a custom vLLM endpoint that exposes embeddings. See vLLM documentation for details.

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration classes
│   ├── vllm_client.py         # vLLM API client
│   ├── temperature_adaptation.py  # Temperature adaptation logic
│   ├── decoding_strategies.py # Contrastive and base decoders
│   ├── generator.py           # Main generator orchestrator
│   └── utils.py               # Utility functions
├── main.py                    # Batch processing script
├── example_single.py          # Single prompt example
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{akyash2025decortlruntimedecodingframework,
      title={DecoRTL: A Run-time Decoding Framework for RTL Code Generation with LLMs}, 
      author={Mohammad Akyash and Kimia Azar and Hadi Kamali},
      year={2025},
      eprint={2507.02226},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2507.02226}, 
}
```

