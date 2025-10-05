# Transformer From Scratch - NumPy Implementation

This repository contains a complete implementation of a GPT-style Transformer model built from scratch using only NumPy, as part of an individual assignment to understand the core architecture of Transformers.

## Developer

| Name | NIM |
|------|-----|
| Muhammad Luthfi Attaqi | 22/496427/TK/54387 |

## Features

### Core Architecture
- **Token Embedding**: Converts input tokens to dense vector representations
- **Positional Encoding**: Sinusoidal positional encoding (standard)
- **Scaled Dot-Product Attention**: Core attention mechanism with softmax normalization
- **Multi-Head Attention**: Parallel attention heads with Q, K, V projections
- **Feed-Forward Network**: Two-layer MLP with GELU activation
- **Residual Connections + Layer Normalization**: Pre-norm architecture for stable training
- **Causal Masking**: Prevents access to future tokens (autoregressive generation)
- **Output Layer**: Projects to vocabulary size with softmax distribution

### Bonus Features
- **Attention Visualization**: Interactive heatmaps and statistical analysis of attention patterns
- **Weight Tying**: Shared weights between embedding and output projection layers
- **RoPE Encoding**: Rotary Positional Embedding as an alternative to sinusoidal encoding

## Dependencies

- **NumPy**: Core mathematical operations
- **Matplotlib**: For attention visualization plots
- **Seaborn**: Enhanced heatmap visualizations
- **Jupyter**: Interactive notebook environment
- **Notebook**: Jupyter notebook server

## Setup and Installation

### 1. Clone or download the repository

```bash
git clone <repository-url>
cd transformer-GPT-style-architecture
```

### 2. Create virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Jupyter Notebook (Recommended)

After activating the virtual environment:

```bash
jupyter notebook
```

This will open Jupyter in your browser. Then:
1. Open `transformer.ipynb`
2. Run cells sequentially using `Shift + Enter`
3. Or run all cells: `Cell → Run All`

The notebook includes:
- Interactive code cells for each component
- Markdown documentation and explanations
- Core architecture tests
- Bonus feature tests (RoPE, Weight Tying, Attention Visualization)
- Comprehensive test results


### Option 2: Run Python Script

Alternatively, you can run the original Python script:

```bash
python transformer.py
```

### Option 3: Open via Google Colab

[Google Colabs](https://colab.research.google.com/drive/1Z20SHd1Mx2aEJW8DyTdpwGgcepirhbB9?usp=sharing) Link

## File Structure

- `transformer.ipynb`: **Main implementation** - Interactive Jupyter notebook with all components
- `transformer.py`: Original Python script version
- `requirements.txt`: Project dependencies
- `README.md`: This documentation file
- `venv/`: Virtual environment (created during setup)
- `.gitignore`: Git ignore configuration
