# Llama DNA Model Converter

Convert dnotitia/llama-dna-1.0-8b-instruct model using UV environment.

## Usage

```bash
# Install dependencies
uv sync

# Run conversion
uv run convert-model

# With 8bit quantization
uv run convert-model --use-8bit

# Check system only
uv run convert-model --check-only
