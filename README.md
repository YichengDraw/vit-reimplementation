# Vision Transformer Reimplementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

From-scratch ViT-B/16 training in PyTorch, plus paper-style visualization reproduction for the original Vision Transformer figures.

- Best reported ImageNet top-1 accuracy: `75.14%`
- Includes training code, figure reproduction code, sample images, and generated outputs

![Attention rollout preview](vit_figures/fig6_attention_rollout.png)

This repository contains two closely related parts:

1. A from-scratch PyTorch implementation of a ViT-B/16-style image classifier for ImageNet-style training.
2. A visualization pipeline that reproduces key figures from the original Vision Transformer paper, *An Image is Worth 16x16 Words*.

The code is aimed at learning, experimentation, and paper reproduction rather than being a fully packaged training framework. The training script keeps the full model and training recipe in one file for readability, while the visualization script focuses on producing paper-style outputs with pretrained checkpoints.

## Highlights

- From-scratch ViT implementation in `vit.py`
- ImageNet-style training recipe with mixup, cutmix, label smoothing, stochastic depth, AMP, gradient accumulation, AdamW, warmup cosine decay, checkpoint resume, and early stopping
- Reported best ImageNet top-1 accuracy: `75.14%`
- Reproduction of four paper-inspired visualizations:
  - Figure 7 Left: patch embedding filters
  - Figure 7 Center: position embedding similarity
  - Figure 7 Right: mean attention distance
  - Figure 6: attention rollout from the output token to the input space
- Included sample images and generated figure outputs

## Repository Structure

```text
vision_transformer/
├── README.md
├── requirements.txt
├── vit.py
├── sample_images/
│   ├── README.md
│   ├── airplane.png
│   ├── dog.png
│   └── snake.png
└── vit_figures/
    ├── vit_visualizations.py
    ├── fig6_attention_rollout.png
    ├── fig7_center_position_similarity.png
    ├── fig7_left_embedding_filters.png
    └── fig7_right_attention_distance.png
```

## Installation

Recommended environment:

- Python 3.10+
- PyTorch 2.x
- CUDA-capable GPU for practical training speed

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### Reproduce the ViT Paper Figures

Run:

```bash
python vit_figures/vit_visualizations.py
```

What this script does:

- Loads local sample images from `sample_images/`
- Downloads pretrained ViT checkpoints on the first run through `timm` and `transformers`
- Saves the generated figures into `vit_figures/`

Expected outputs:

- `vit_figures/fig7_left_embedding_filters.png`
- `vit_figures/fig7_center_position_similarity.png`
- `vit_figures/fig7_right_attention_distance.png`
- `vit_figures/fig6_attention_rollout.png`

### Train the From-Scratch ViT

The training script uses `torchvision.datasets.ImageFolder`, so your dataset should follow the usual class-folder structure.

Default expected layout:

```text
data/
└── imagenet/
    ├── train/
    │   ├── class_001/
    │   ├── class_002/
    │   └── ...
    └── val/
        ├── class_001/
        ├── class_002/
        └── ...
```

You can either:

- place the dataset at `data/imagenet/train` and `data/imagenet/val`, or
- override the paths with environment variables

Optional environment variables:

- `IMAGENET_TRAIN_DIR`
- `IMAGENET_VAL_DIR`
- `VIT_CHECKPOINT_PATH`
- `VIT_TENSORBOARD_DIR`

Run training:

```bash
python vit.py
```

By default, training artifacts are written to:

- `outputs/checkpoints/vit_imagenet_best.pth`
- `outputs/runs/vit_imagenet/`

To inspect TensorBoard logs:

```bash
tensorboard --logdir outputs/runs
```

## Reported Training Result

The current `vit.py` training recipe reports:

- Best top-1 accuracy: `75.14%`
- Model family: ViT-B/16-style classifier
- Input resolution: `224 x 224`

This number is important because `vit.py` is not just a bare-bones architecture demo. It includes a full training recipe with multiple performance-oriented and stability-oriented techniques, and the reported result reflects the combined effect of those choices.

## Architecture

### Model Architecture (`vit.py`)

The classifier follows the standard ViT encoder design:

```text
image
  -> patch embedding (Conv2d with kernel=stride=patch_size)
  -> flatten to patch tokens
  -> prepend CLS token
  -> add learned positional embeddings
  -> Transformer encoder blocks x 12
  -> LayerNorm
  -> CLS token projection head
  -> logits
```

Main components:

- `PatchEmbed`: converts the input image into non-overlapping patch tokens with a convolutional projection
- `Attention`: multi-head self-attention implemented with `qkv` projection and PyTorch scaled dot-product attention
- `MLP`: two-layer feed-forward network with GELU and dropout
- `DropPath`: stochastic depth for residual branch regularization
- `Block`: pre-norm Transformer block with residual attention and residual MLP
- `ViT`: full encoder with CLS token, learned positional embeddings, stacked blocks, final normalization, and classification head

### Training Pipeline (`vit.py`)

The training recipe includes:

- `ImageFolder`-based data loading
- Training augmentation with random resized crop, horizontal flip, RandAugment, normalization, and random erasing
- Validation preprocessing with resize and center crop
- Mixup and CutMix sampling inside the training loop
- Label smoothing loss
- Automatic mixed precision through `autocast` and `GradScaler`
- Gradient accumulation and gradient clipping
- AdamW optimizer
- Warmup + cosine learning-rate scheduling
- Checkpoint resume logic
- Early stopping based on validation top-1 improvement

### Training Tricks Used in `vit.py`

The main techniques that improve optimization, regularization, or training stability are:

- Patch embedding with a `16 x 16` stride-convolution projection, matching the ViT-B/16 tokenization pattern
- Learned CLS token and learned positional embeddings
- Stochastic depth through `DropPath` with a non-zero drop-path schedule across Transformer blocks
- RandAugment and Random Erasing in the image augmentation pipeline
- Mixup and CutMix during training for stronger regularization
- Label smoothing to reduce overconfidence
- AdamW with decoupled weight decay
- Warmup followed by cosine learning-rate decay
- Automatic mixed precision with `autocast` and `GradScaler`
- Gradient accumulation to support an effectively larger batch size
- Gradient clipping to reduce unstable optimizer steps
- Checkpoint resume support so long runs can continue without restarting from scratch
- Early stopping and best-checkpoint saving based on validation top-1

In short, `vit.py` combines architecture-level choices, data-level augmentation, loss regularization, optimizer scheduling, and mixed-precision training. Those details are a major reason the implementation is useful for users who want more than a minimal ViT forward pass.

### Visualization Pipeline (`vit_figures/vit_visualizations.py`)

The figure script is intentionally separate from the training script because it uses pretrained public checkpoints and figure-specific logic.

Figure generation flow:

1. Resolve fixed local sample images from `sample_images/`
2. Load pretrained ViT checkpoints from `timm` or Hugging Face
3. Run figure-specific analysis
4. Save rendered outputs to `vit_figures/`

Per-figure logic:

- Figure 7 Left:
  Extract patch embedding weights, run PCA, and render paper-style grayscale and RGB filter tiles.
- Figure 7 Center:
  Compute cosine similarity between learned position embeddings for all patch positions.
- Figure 7 Right:
  Register hooks on the attention modules, reconstruct attention weights, and compute mean patch-to-patch attention distance per head and per layer.
- Figure 6:
  Use a Hugging Face ViT with `output_attentions=True`, apply attention rollout across layers, upsample the rollout map, and overlay it on the original sample images.

## Implementation Notes

### Why the Figure Script Uses Multiple Checkpoints

The figure reproduction script does not force a single checkpoint for every panel. Different pretrained models are chosen when they produce results that are closer to the visual character of the original paper:

- `google/vit-large-patch32-224-in21k` for embedding filters
- `google/vit-large-patch16-224-in21k` for attention rollout
- `vit_base_patch16_224_in21k` from `timm` for position similarity and attention distance

This is an intentional practical choice for figure quality, not an attempt to claim exact paper recreation from one unified checkpoint.

### Design Priorities

- Keep the training implementation readable and compact
- Keep figure-generation logic explicit rather than over-abstracted
- Preserve paper-aligned outputs where visual details matter
- Use local sample images so the visualization output is reproducible across runs

## Generated Figure Preview

### Figure 6: Attention Rollout

![Figure 6 attention rollout](vit_figures/fig6_attention_rollout.png)

### Figure 7 Left: Embedding Filters

![Figure 7 left embedding filters](vit_figures/fig7_left_embedding_filters.png)

### Figure 7 Center: Position Embedding Similarity

![Figure 7 center position embedding similarity](vit_figures/fig7_center_position_similarity.png)

### Figure 7 Right: Mean Attention Distance

![Figure 7 right mean attention distance](vit_figures/fig7_right_attention_distance.png)

## Known Limitations

- `vit.py` is a single-file research and learning implementation, not a modular training framework
- The training script is configured through code and environment variables rather than a CLI or YAML config system
- The current training path assumes ImageNet-style folder datasets via `ImageFolder`
- Figure reproduction depends on downloading pretrained weights the first time you run it
- The visualizations are intended to be paper-style reproductions, not guaranteed pixel-exact copies of the original paper figures
- The repository currently does not include automated tests

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- Abnar and Zuidema, *Quantifying Attention Flow in Transformers*

## License

This project is released under the MIT License. See `LICENSE` for details.
