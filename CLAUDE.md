# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wan2.1 is an open-source video generation foundation model by Alibaba's Wan Team. It supports multiple tasks: Text-to-Video (T2V), Image-to-Video (I2V), First-Last-Frame-to-Video (FLF2V), Text-to-Image (T2I), and Video Creation/Editing (VACE).

Two model sizes are available:
- **14B**: Full-featured, supports 480P and 720P
- **1.3B**: Lightweight, 480P only, runs on consumer GPUs (8GB+ VRAM)

## Common Commands

### Installation
```bash
pip install -r requirements.txt
pip install .              # Standard install
pip install .[dev]         # With dev tools (pytest, black, flake8, isort, mypy)
```

### Running Inference
```bash
# Text-to-Video (single GPU)
python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "..."

# Text-to-Video (multi-GPU with FSDP)
torchrun --nproc_per_node=8 generate.py --task t2v-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --ckpt_dir ./Wan2.1-T2V-14B --prompt "..."

# Image-to-Video
python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --prompt "..."

# Low VRAM mode (for consumer GPUs)
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --prompt "..."
```

### Testing & Formatting
```bash
pytest tests/                    # Run tests
make format                      # Format with isort + yapf
black . && isort .               # Alternative formatting
bash tests/test.sh <model_dir> <gpu_count>  # Full model test suite
```

## Architecture Overview

### Core Package Structure (`wan/`)
```
wan/
├── text2video.py          # WanT2V - main T2V generation class
├── image2video.py         # WanI2V - I2V generation class
├── first_last_frame2video.py  # WanFLF2V - frame interpolation
├── vace.py                # WanVace, WanVaceMP - video editing
├── configs/               # Model configurations (14B, 1.3B variants)
├── modules/               # Core neural network components
│   ├── model.py          # WanModel - Diffusion Transformer (DiT)
│   ├── vae.py            # WanVAE - 3D Causal VAE
│   ├── attention.py      # Flash attention implementation
│   ├── t5.py             # T5 text encoder
│   └── clip.py           # CLIP vision encoder
├── utils/                 # Utilities
│   ├── fm_solvers.py     # Flow Matching solvers (DPM++, UniPC)
│   └── prompt_extend.py  # Prompt expansion (DashScope/Qwen)
└── distributed/           # Multi-GPU support (FSDP, xDiT)
```

### Generation Pipeline
1. **Text Encoding**: T5 encoder (512 tokens) for multilingual prompts
2. **Image Encoding**: CLIP for I2V/FLF2V tasks
3. **VAE Encoding**: Compress input to latent space
4. **DiT Sampling**: Diffusion Transformer with Flow Matching
5. **VAE Decoding**: Reconstruct video/image from latents

### Key Entry Point: `generate.py`
Command-line interface for all tasks. Important arguments:
- `--task`: t2v-14B, t2v-1.3B, i2v-14B, flf2v-14B, t2i-14B, vace-1.3B, vace-14B
- `--size`: Output resolution (1280*720, 720*1280, 832*480, 480*832, 1024*1024)
- `--ckpt_dir`: Model checkpoint directory (required)
- `--offload_model`: CPU offloading for low VRAM
- `--t5_cpu`: Place T5 encoder on CPU
- `--dit_fsdp`, `--t5_fsdp`: Enable FSDP for multi-GPU
- `--ulysses_size`, `--ring_size`: xDiT parallelization strategies

### Configuration System (`wan/configs/`)
- `WAN_CONFIGS`: Maps task names to model configurations
- `SUPPORTED_SIZES`: Valid resolutions per task
- Model dimensions: 14B (5120-dim, 40 layers), 1.3B (1536-dim, 30 layers)

### Gradio UIs (`gradio/`)
Web interfaces for each task. Run from `gradio/` directory:
```bash
python t2v_14B_singleGPU.py --ckpt_dir ./Wan2.1-T2V-14B
```

## Code Style

- YAPF formatter (config in `.style.yapf`, 80 char lines)
- isort for imports (black profile)
- Python 3.10 required
- Type hints with strict mypy checking
