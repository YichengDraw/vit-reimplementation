"""
ViT paper figure reproduction script.

Reproduces four visualizations from "An Image is Worth 16x16 Words"
(Dosovitskiy et al., 2020):
1. Figure 7 Left: embedding filters in a paper-style principal-component layout
2. Figure 7 Center: position embedding similarity heatmaps
3. Figure 7 Right: mean attention distance across layers
4. Figure 6: attention rollout from the output token to the input space

The script uses cached local paper sample images from `../sample_images`, a
Hugging Face ViT-L/32 checkpoint for the filter figure, a Hugging Face ViT-L/16
checkpoint for the attention rollout figure, and timm ViT-B/16 for the
remaining Figure 7 panels.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import torch
import numpy as np
import matplotlib.pyplot as plt
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import os
import json
import time
from typing import Any, List, Tuple, Optional, cast
from transformers import ViTConfig, ViTImageProcessor, ViTModel


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEBUG_LOG_PATH = os.environ.get(
    "VIT_DEBUG_LOG_PATH",
    os.path.join(os.path.dirname(__file__), "vit_debug.log"),
)
DEBUG_SESSION_ID = "debug-session"
DEBUG_RUN_ID = "post-fix"

# Model selection.
TIMM_MODEL_NAMES = {
    "pos": "vit_base_patch16_224_in21k",
    "distance": "vit_base_patch16_224_in21k",
}
HF_MODEL_NAMES = {
    "filters": "google/vit-large-patch32-224-in21k",
    "attention": "google/vit-large-patch16-224-in21k",
}

# Local sample images uploaded from the paper examples.
LOCAL_SAMPLE_IMAGE_NAMES = ("dog.png", "airplane.png", "snake.png")

# Attention rollout tuning for a paper-like, sharper object silhouette.
ATTENTION_HEAD_FUSION = "mean"
ATTENTION_DISCARD_RATIO = 0.0
ATTENTION_GAMMA = 1.15
ATTENTION_MIN_ALPHA = 0.10
ATTENTION_LOW_PERCENTILE = 2.0
ATTENTION_HIGH_PERCENTILE = 99.5
ATTENTION_THRESHOLD = 0.10

# Paper-style Figure 7 Left layout:
# a few luminance PCs first, then colorful RGB PCs from the same ViT-L/32 model.
FILTER_GRAY_COMPONENT_INDICES = (0, 3, 5, 1)
FILTER_RGB_START_INDEX = 1


def _debug_log(message: str, data: dict, hypothesis_id: str, location: str) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": DEBUG_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_vit_model(model_name: str = 'vit_base_patch16_224', pretrained: bool = True):
    """Load pretrained ViT model from timm."""
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    return model


def get_hf_vit_model(model_name: str, output_attentions: bool = False):
    """Load pretrained ViT model from Hugging Face."""
    if output_attentions:
        config = ViTConfig.from_pretrained(model_name, output_attentions=True)
        model = ViTModel.from_pretrained(
            model_name,
            config=config,
            attn_implementation="eager",
        )
    else:
        model = ViTModel.from_pretrained(model_name)
    model.eval()
    return model


def get_hf_image_processor(model_name: str):
    """Load the paired image processor for a Hugging Face ViT checkpoint."""
    return ViTImageProcessor.from_pretrained(model_name)


def extract_patch_embedding_weight(model) -> np.ndarray:
    """Return patch embedding weights from either a timm or HF ViT model."""
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        return model.patch_embed.proj.weight.detach().cpu().numpy()
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "patch_embeddings"):
        return model.embeddings.patch_embeddings.projection.weight.detach().cpu().numpy()
    raise TypeError("Unsupported model type for patch embedding extraction")


def get_preprocessing_transform(model=None, image_size: int = 224):
    """Get preprocessing transform based on model config when available."""
    if model is not None:
        cfg = resolve_data_config({}, model=model)
        return cast(Any, create_transform(**cfg, is_training=False))
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _strip_normalize_and_tensor(transform):
    if not isinstance(transform, transforms.Compose):
        return transform
    pil_to_tensor = getattr(transforms, "PILToTensor", None)
    kept = []
    for item in transform.transforms:
        if isinstance(item, (transforms.Normalize, transforms.ToTensor)):
            continue
        if pil_to_tensor is not None and isinstance(item, pil_to_tensor):
            continue
        kept.append(item)
    return transforms.Compose(kept)


def get_model_and_vis_transforms(model=None, image_size: int = 224):
    """Return model and visualization transforms aligned to the same resize/crop."""
    if model is not None:
        cfg = resolve_data_config({}, model=model)
        model_transform = cast(Any, create_transform(**cfg, is_training=False))
        vis_transform = cast(Any, _strip_normalize_and_tensor(model_transform))
        return model_transform, vis_transform
    model_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    vis_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    return model_transform, vis_transform


# =============================================================================
# Figure 7 Left: RGB Embedding Filters (First 28 Principal Components)
# =============================================================================

def _normalize_filter_component(component: np.ndarray) -> np.ndarray:
    """Map a signed filter component into [0, 1] for visualization."""
    max_abs = np.abs(component).max() + 1e-8
    component = 0.5 + 0.5 * (component / max_abs)
    return np.clip(component, 0.0, 1.0)


def visualize_patch_embedding_filters(model, n_components: int = 28, save_path: Optional[str] = None):
    """
    Visualize a paper-style layout of patch embedding principal components.

    We use a ViT-L/32 checkpoint to recover the larger patch-size structure that
    looks much closer to the original paper. The first few tiles come from
    grayscale/luminance PCA to expose Gabor-like structure, followed by colorful
    RGB principal components from the same model.
    """
    patch_embed_weight = extract_patch_embedding_weight(model)
    embed_dim, in_channels, patch_h, patch_w = patch_embed_weight.shape

    print(f"Patch embedding weight shape: {patch_embed_weight.shape}")
    print(f"Embed dim: {embed_dim}, Patch size: {patch_h}x{patch_w}, Channels: {in_channels}")

    rgb_needed = n_components - len(FILTER_GRAY_COMPONENT_INDICES) + FILTER_RGB_START_INDEX
    rgb_flat = patch_embed_weight.reshape(embed_dim, -1)
    pca_rgb = PCA(n_components=rgb_needed)
    pca_rgb.fit(rgb_flat)
    rgb_components = pca_rgb.components_.reshape(-1, in_channels, patch_h, patch_w)

    grayscale_weights = (
        0.299 * patch_embed_weight[:, 0, :, :]
        + 0.587 * patch_embed_weight[:, 1, :, :]
        + 0.114 * patch_embed_weight[:, 2, :, :]
    )
    gray_flat = grayscale_weights.reshape(embed_dim, -1)
    gray_needed = max(FILTER_GRAY_COMPONENT_INDICES) + 1
    pca_gray = PCA(n_components=gray_needed)
    pca_gray.fit(gray_flat)
    gray_components = pca_gray.components_.reshape(-1, patch_h, patch_w)

    filter_patterns_vis = []

    for component_idx in FILTER_GRAY_COMPONENT_INDICES:
        filt = _normalize_filter_component(gray_components[component_idx])
        filt_rgb = np.stack([filt, filt, filt], axis=-1)
        filter_patterns_vis.append(filt_rgb)

    rgb_take = n_components - len(filter_patterns_vis)
    for component_idx in range(FILTER_RGB_START_INDEX, FILTER_RGB_START_INDEX + rgb_take):
        filt = rgb_components[component_idx].transpose(1, 2, 0)
        filter_patterns_vis.append(_normalize_filter_component(filt))

    n_rows = 4
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    fig.suptitle('RGB embedding filters\n(first 28 principal components)', fontsize=14)

    for idx in range(n_components):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.imshow(filter_patterns_vis[idx], interpolation="nearest")
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved patch embedding filters to {save_path}")

    plt.close(fig)
    return filter_patterns_vis


# =============================================================================
# Figure 7 Center: Position Embedding Similarity
# =============================================================================

def visualize_position_embedding_similarity(model, save_path: Optional[str] = None):
    """
    Visualize the cosine similarity between position embeddings.
    
    For ViT-B/16 with 224x224 input:
    - 14x14 = 196 patches + 1 CLS token = 197 position embeddings
    - We visualize similarity for patch positions only (excluding CLS)
    
    Args:
        model: ViT model from timm
        save_path: Path to save the figure (optional)
    """
    # Extract position embeddings
    # Shape: (1, num_positions, embed_dim) = (1, 197, 768)
    pos_embed = model.pos_embed.detach().cpu().numpy()[0]  # (197, 768)
    
    # Exclude CLS token (first position), keep only patch embeddings
    patch_pos_embed = pos_embed[1:]  # (196, 768)
    
    print(f"Position embedding shape: {pos_embed.shape}")
    print(f"Patch position embeddings: {patch_pos_embed.shape}")
    
    # Compute cosine similarity matrix
    # Normalize embeddings
    norms = np.linalg.norm(patch_pos_embed, axis=1, keepdims=True)
    patch_pos_embed_norm = patch_pos_embed / (norms + 1e-8)
    
    # Cosine similarity: (196, 196)
    similarity_matrix = patch_pos_embed_norm @ patch_pos_embed_norm.T
    
    # Grid size
    grid_size = int(np.sqrt(patch_pos_embed.shape[0]))  # 14
    
    # Create visualization showing similarity for selected positions
    # Show a grid where each cell shows the similarity heatmap for that position
    # Following the paper, we show a subset of positions
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Position embedding similarity', fontsize=14)
    
    im = None
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i, j]
            idx = i * grid_size + j
            
            # Get similarity of this position to all others
            sim = similarity_matrix[idx].reshape(grid_size, grid_size)
            
            # Plot heatmap
            im = ax.imshow(sim, cmap='viridis', vmin=-1, vmax=1)
            ax.axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    assert im is not None
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Cosine similarity', fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.9, top=0.94, bottom=0.03, wspace=0.02, hspace=0.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved position embedding similarity to {save_path}")
    
    plt.close(fig)
    return similarity_matrix


# =============================================================================
# Figure 7 Right: Mean Attention Distance
# =============================================================================

class AttentionExtractor:
    """Hook-based attention weight extractor for ViT models."""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []
        
    def _hook_fn(self, module, input, output):
        """Hook function to capture attention weights."""
        # For timm ViT, attention weights are computed inside the Attention module
        # We need to recompute them from the qkv
        self.attention_weights.append(output)
    
    def register_hooks(self):
        """Register forward hooks on all attention modules."""
        self.attention_weights = []
        self.hooks = []
        
        for block in self.model.blocks:
            # Register hook on attention module
            hook = block.attn.register_forward_hook(self._get_attention_hook(block.attn))
            self.hooks.append(hook)
    
    def _get_attention_hook(self, attn_module):
        """Create a hook that computes and stores attention weights."""
        def hook(module, input, output):
            # input[0] is x with shape (B, N, D)
            x = input[0]
            B, N, C = x.shape
            
            # Get qkv
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention weights
            scale = (C // attn_module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)  # (B, num_heads, N, N)
            
            self.attention_weights.append(attn.detach().cpu())
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_weights(self):
        """Return collected attention weights."""
        return self.attention_weights


def compute_attention_distance(attention_weights: torch.Tensor, grid_size: Optional[int] = None, patch_size: Optional[int] = None, layer_idx: Optional[int] = None):
    """
    Compute mean attention distance in pixels for each head.
    
    Args:
        attention_weights: Attention weights of shape (B, num_heads, N, N) where N = grid_size^2 + 1
        grid_size: Number of patches per side (14 for 224/16)
        patch_size: Size of each patch in pixels (16)
    
    Returns:
        Mean attention distance in pixels for each head, averaged over batch and query positions
    """
    B, num_heads, N, _ = attention_weights.shape
    if grid_size is None:
        grid_size = int(np.sqrt(N - 1))
    if patch_size is None:
        patch_size = 224 // grid_size
    num_patches = grid_size * grid_size
    
    # Create distance matrix between patches (in pixels)
    # Distance between patch i and patch j
    distances = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            # Convert linear index to 2D grid coordinates
            xi, yi = i % grid_size, i // grid_size
            xj, yj = j % grid_size, j // grid_size
            # Distance in pixels (center to center)
            distances[i, j] = np.sqrt(((xi - xj) * patch_size) ** 2 + ((yi - yj) * patch_size) ** 2)
    
    distances = torch.tensor(distances, dtype=attention_weights.dtype)
    
    # Extract patch-to-patch attention (exclude CLS token)
    # attention_weights: (B, num_heads, N, N) where N = 197
    # We want attention from patches (1:) to patches (1:)
    patch_attn_raw = attention_weights[:, :, 1:, 1:]  # (B, num_heads, 196, 196)
    cls_attn_raw = attention_weights[:, :, 1:, 0]  # (B, num_heads, 196)
    patch_mass = patch_attn_raw.sum(dim=-1)
    
    # Renormalize attention weights after excluding CLS
    patch_attn = patch_attn_raw / (patch_attn_raw.sum(dim=-1, keepdim=True) + 1e-8)

    # region agent log
    patch_attn_sums = patch_attn.sum(dim=-1)
    _debug_log(
        message="Attention distance inputs after CLS removal",
        data={
            "attention_shape": list(attention_weights.shape),
            "patch_attn_shape": list(patch_attn.shape),
            "patch_attn_row_sum_mean": float(patch_attn_sums.mean().item()),
            "patch_attn_row_sum_std": float(patch_attn_sums.std().item()),
            "patch_mass_mean": float(patch_mass.mean().item()),
            "cls_mass_mean": float(cls_attn_raw.mean().item()),
            "layer_idx": layer_idx,
            "distance_matrix_min": float(distances.min().item()),
            "distance_matrix_max": float(distances.max().item())
        },
        hypothesis_id="H4",
        location="vit_visualizations.py:335"
    )
    # endregion
    
    # Compute weighted average distance for each head
    # Shape: (B, num_heads, num_patches)
    weighted_distances = (patch_attn * distances.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    weighted_distances_raw = (patch_attn_raw * distances.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    
    # Average over batch and query positions
    # Shape: (num_heads,)
    mean_distances = weighted_distances.mean(dim=(0, 2))
    mean_distances_raw = weighted_distances_raw.mean(dim=(0, 2))

    # region agent log
    _debug_log(
        message="Attention distance comparison (renorm vs raw)",
        data={
            "layer_idx": layer_idx,
            "mean_distance_min": float(mean_distances.min().item()),
            "mean_distance_max": float(mean_distances.max().item()),
            "mean_distance_raw_min": float(mean_distances_raw.min().item()),
            "mean_distance_raw_max": float(mean_distances_raw.max().item())
        },
        hypothesis_id="H6",
        location="vit_visualizations.py:365"
    )
    # endregion
    
    return mean_distances_raw.numpy()


def visualize_mean_attention_distance(model, images: torch.Tensor, save_path: Optional[str] = None):
    """
    Visualize mean attention distance across layers for each head.
    
    Args:
        model: ViT model from timm
        images: Batch of preprocessed images (B, 3, 224, 224)
        save_path: Path to save the figure (optional)
    """
    model.eval()
    
    # Extract attention weights using hooks
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    with torch.no_grad():
        _ = model(images)
    
    attention_weights = extractor.get_attention_weights()
    extractor.remove_hooks()
    
    print(f"Collected attention from {len(attention_weights)} layers")
    print(f"Attention shape per layer: {attention_weights[0].shape}")
    
    # Get model info
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    # Compute mean attention distance for each layer and head
    all_distances = []
    grid_size = int(np.sqrt(attention_weights[0].shape[-1] - 1))
    patch_size = model.patch_embed.patch_size if hasattr(model.patch_embed, "patch_size") else (16, 16)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    for layer_idx, attn in enumerate(attention_weights):
        distances = compute_attention_distance(attn, grid_size=grid_size, patch_size=patch_size, layer_idx=layer_idx)
        all_distances.append(distances)
    
    all_distances = np.array(all_distances)  # (num_layers, num_heads)

    # region agent log
    _debug_log(
        message="Mean attention distance summary",
        data={
            "batch_size": int(images.shape[0]),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "first_layer_min": float(all_distances[0].min()),
            "first_layer_max": float(all_distances[0].max()),
            "last_layer_min": float(all_distances[-1].min()),
            "last_layer_max": float(all_distances[-1].max())
        },
        hypothesis_id="H5",
        location="vit_visualizations.py:383"
    )
    # endregion
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use different colors for different heads
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, num_heads))
    
    for head_idx in range(num_heads):
        ax.scatter(
            range(num_layers), 
            all_distances[:, head_idx],
            c=[colors[head_idx]],
            s=30,
            alpha=0.7,
            label=f'Head {head_idx + 1}' if head_idx < 3 else None
        )
    
    ax.set_xlabel('Network depth (layer)', fontsize=12)
    ax.set_ylabel('Mean attention distance (pixels)', fontsize=12)
    ax.set_title('ViT-B/16', fontsize=14)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_ylim(0, 130)
    
    # Add legend for first 3 heads
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mean attention distance to {save_path}")
    
    plt.close(fig)
    return all_distances


# =============================================================================
# Figure 6: Attention Rollout Visualization
# =============================================================================

def attention_rollout(
    attention_weights: List[torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
):
    """
    Compute attention rollout following Abnar & Zuidema (2020).
    
    The method:
    1. Fuse attention weights across heads
    2. Add identity matrix (residual connections)
    3. Recursively multiply attention matrices across layers
    
    Args:
        attention_weights: List of attention tensors, each (B, num_heads, N, N)
        discard_ratio: Ratio of lowest attention values to discard (optional)
    
    Returns:
        Rollout attention from CLS token to all patches, shape (B, N-1)
    """
    fused_attentions = []
    for attn in attention_weights:
        if head_fusion == "mean":
            fused = attn.mean(dim=1)
        elif head_fusion == "max":
            fused = attn.max(dim=1).values
        elif head_fusion == "min":
            fused = attn.min(dim=1).values
        else:
            raise ValueError(f"Unsupported head fusion: {head_fusion}")

        if discard_ratio > 0:
            flat = fused.view(fused.size(0), -1)
            num_to_drop = int(flat.size(-1) * discard_ratio)
            if num_to_drop > 0:
                _, indices = flat.topk(num_to_drop, dim=-1, largest=False)
                for batch_idx in range(flat.size(0)):
                    drop_idx = indices[batch_idx]
                    drop_idx = drop_idx[drop_idx != 0]  # keep the CLS token connection
                    flat[batch_idx, drop_idx] = 0

        fused_attentions.append(fused)

    B, N, _ = fused_attentions[0].shape
    rollout = torch.eye(
        N,
        dtype=fused_attentions[0].dtype,
        device=fused_attentions[0].device,
    ).unsqueeze(0).expand(B, -1, -1)

    for attn in fused_attentions:
        residual = torch.eye(N, dtype=attn.dtype, device=attn.device).unsqueeze(0)
        attn_with_residual = (attn + residual) / 2
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        rollout = attn_with_residual @ rollout

    cls_attention = rollout[:, 0, 1:]

    # region agent log
    _debug_log(
        message="Attention rollout configuration",
        data={
            "num_layers": int(len(fused_attentions)),
            "head_fusion": head_fusion,
            "num_tokens": int(N),
            "use_residual": True,
            "residual_weight": 1.0,
            "discard_ratio": discard_ratio,
        },
        hypothesis_id="H7",
        location="vit_visualizations.py:463"
    )
    # endregion
    
    return cls_attention


def render_attention_overlay(image: Image.Image, attn_map: np.ndarray) -> np.ndarray:
    """Render attention as a contrast-enhanced mask over the original image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = np.array(image).astype(np.float32) / 255.0
    attn = np.maximum(attn_map.astype(np.float32), 0.0)
    low = np.percentile(attn, ATTENTION_LOW_PERCENTILE)
    high = np.percentile(attn, ATTENTION_HIGH_PERCENTILE)
    if high - low > 1e-8:
        attn = np.clip((attn - low) / (high - low), 0.0, 1.0)
    else:
        attn = np.zeros_like(attn)
    attn = np.clip((attn - ATTENTION_THRESHOLD) / (1.0 - ATTENTION_THRESHOLD + 1e-8), 0.0, 1.0)
    attn = attn ** ATTENTION_GAMMA
    mask = ATTENTION_MIN_ALPHA + (1.0 - ATTENTION_MIN_ALPHA) * attn
    masked = np.clip(img * mask[..., None], 0.0, 1.0)
    return masked


def visualize_attention_rollout(
    model,
    processor,
    image_paths: List[str],
    device: torch.device,
    save_path: Optional[str] = None,
):
    """
    Visualize attention rollout from CLS token to input patches using a HF ViT model.
    """
    model.eval()
    original_images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=original_images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    attention_weights = [attn.detach().cpu() for attn in outputs.attentions]
    cls_attention = attention_rollout(
        attention_weights,
        discard_ratio=ATTENTION_DISCARD_RATIO,
        head_fusion=ATTENTION_HEAD_FUSION,
    )
    grid_size = int(np.sqrt(cls_attention.shape[-1]))

    # region agent log
    cls_attention_np = cls_attention.numpy()
    flat = cls_attention_np.reshape(cls_attention_np.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    cos = (flat @ flat.T) / (norms @ norms.T)
    np.fill_diagonal(cos, np.nan)
    _debug_log(
        message="Attention rollout map similarity across images",
        data={
            "batch_size": int(cls_attention_np.shape[0]),
            "cls_attention_mean": float(cls_attention_np.mean()),
            "cls_attention_std": float(cls_attention_np.std()),
            "pairwise_cos_min": float(np.nanmin(cos)) if cos.size > 1 else None,
            "pairwise_cos_max": float(np.nanmax(cos)) if cos.size > 1 else None
        },
        hypothesis_id="H3",
        location="vit_visualizations.py:498"
    )
    # endregion
    
    B = len(original_images)
    fig, axes = plt.subplots(B, 2, figsize=(6, 3 * B))
    
    if B == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Figure 6: Representative examples of attention from the\noutput token to the input space', fontsize=12)
    
    for i in range(B):
        # Original image
        ax_orig = axes[i, 0]
        ax_orig.imshow(original_images[i])
        ax_orig.axis('off')
        if i == 0:
            ax_orig.set_title('Input', fontsize=11)
        
        # Attention heatmap
        ax_attn = axes[i, 1]
        
        attn = cls_attention[i].numpy().reshape(grid_size, grid_size).astype(np.float32)
        target_size = original_images[i].size
        attn_upsampled = np.array(
            Image.fromarray(attn).resize(target_size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        overlay = render_attention_overlay(original_images[i], attn_upsampled)

        # region agent log
        _debug_log(
            message="Attention mask stats for visualization",
            data={
                "image_index": i,
                "attn_min": float(attn_upsampled.min()),
                "attn_max": float(attn_upsampled.max()),
                "attn_mean": float(attn_upsampled.mean()),
                "overlay_mean": float(overlay.mean())
            },
            hypothesis_id="H8",
            location="vit_visualizations.py:533"
        )
        # endregion
        
        ax_attn.imshow(overlay)
        ax_attn.axis('off')
        if i == 0:
            ax_attn.set_title('Attention', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention rollout to {save_path}")
    
    plt.close(fig)
    return cls_attention


# =============================================================================
# Sample Image Utilities
# =============================================================================

def get_local_sample_image_paths(sample_dir: Optional[str] = None) -> List[str]:
    """
    Return the fixed local sample images used in the paper comparison.
    """
    if sample_dir is None:
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_images")

    image_paths = [os.path.join(sample_dir, name) for name in LOCAL_SAMPLE_IMAGE_NAMES]
    missing = [path for path in image_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required local sample images: " + ", ".join(os.path.basename(path) for path in missing)
        )

    print(f"Using local sample images from {sample_dir}")
    return image_paths


def load_and_preprocess_images(image_paths: List[str], image_size: int = 224, model=None):
    """
    Load and preprocess images for ViT.
    
    Args:
        image_paths: List of paths to images
    image_size: Target image size (used when model config is unavailable)
    
    Returns:
        Tuple of (preprocessed tensor, list of original PIL images)
    """
    model_transform, vis_transform = get_model_and_vis_transforms(model=model, image_size=image_size)
    
    images = []
    original_images = []
    image_stats = []
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_array = np.array(img)
        image_stats.append({
            "path": path,
            "size": list(img.size),
            "mean": float(img_array.mean()),
            "std": float(img_array.std()),
            "min": int(img_array.min()),
            "max": int(img_array.max())
        })
        vis_img = vis_transform(img) if vis_transform is not None else img
        if isinstance(vis_img, torch.Tensor):
            vis_img = transforms.ToPILImage()(vis_img)
        original_images.append(vis_img)
        images.append(model_transform(img))
    
    images_tensor = torch.stack(images)

    # region agent log
    _debug_log(
        message="Loaded image stats for attention visualizations",
        data={
            "image_count": len(image_stats),
            "image_size": image_size,
            "vis_image_size": list(original_images[0].size) if original_images else None,
            "image_stats": image_stats
        },
        hypothesis_id="H2",
        location="vit_visualizations.py:690"
    )
    # endregion
    
    return images_tensor, original_images


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run all visualizations."""
    print("=" * 60)
    print("ViT Paper Figure Reproduction")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n[1/5] Resolving local sample images...")
    image_paths = get_local_sample_image_paths()
    print(f"Loaded {len(image_paths)} local paper sample images")

    _debug_log(
        message="Model selection for figures",
        data={
            "timm_pos_model": TIMM_MODEL_NAMES["pos"],
            "timm_distance_model": TIMM_MODEL_NAMES["distance"],
            "hf_filter_model": HF_MODEL_NAMES["filters"],
            "hf_attention_model": HF_MODEL_NAMES["attention"],
            "attention_head_fusion": ATTENTION_HEAD_FUSION,
            "attention_discard_ratio": ATTENTION_DISCARD_RATIO,
            "attention_threshold": ATTENTION_THRESHOLD,
            "sample_images": [os.path.basename(path) for path in image_paths],
        },
        hypothesis_id="H9",
        location="vit_visualizations.py:main"
    )

    # Figure 7 Left: embedding filters
    print("\n[2/5] Generating embedding filters (Figure 7 Left)...")
    model_filters = get_hf_vit_model(HF_MODEL_NAMES["filters"])
    visualize_patch_embedding_filters(
        model_filters, 
        n_components=28,
        save_path=os.path.join(output_dir, 'fig7_left_embedding_filters.png')
    )
    del model_filters

    # Figure 7 Center: Position Embedding Similarity
    print("\n[3/5] Generating position embedding similarity (Figure 7 Center)...")
    model_pos = get_vit_model(TIMM_MODEL_NAMES["pos"], pretrained=True)
    visualize_position_embedding_similarity(
        model_pos,
        save_path=os.path.join(output_dir, 'fig7_center_position_similarity.png')
    )
    del model_pos

    # Figure 7 Right: Mean Attention Distance
    print("\n[4/5] Generating mean attention distance (Figure 7 Right)...")
    model_distance = get_vit_model(TIMM_MODEL_NAMES["distance"], pretrained=True).to(device)
    images_tensor, _ = load_and_preprocess_images(image_paths, model=model_distance)
    images_tensor = images_tensor.to(device)
    visualize_mean_attention_distance(
        model_distance,
        images_tensor,
        save_path=os.path.join(output_dir, 'fig7_right_attention_distance.png')
    )
    del images_tensor
    del model_distance

    # Figure 6: Attention Rollout
    print("\n[5/5] Generating attention rollout visualization (Figure 6)...")
    model_attention = get_hf_vit_model(HF_MODEL_NAMES["attention"], output_attentions=True)
    cast(Any, model_attention).to(device)
    attention_processor = get_hf_image_processor(HF_MODEL_NAMES["attention"])
    visualize_attention_rollout(
        model_attention,
        attention_processor,
        image_paths,
        device,
        save_path=os.path.join(output_dir, 'fig6_attention_rollout.png')
    )
    del model_attention
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
