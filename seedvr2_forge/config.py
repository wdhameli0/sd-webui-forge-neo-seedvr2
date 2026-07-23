from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SeedVR2Config:
    enabled: bool = True
    dit_model: str = "seedvr2_ema_3b_fp16.safetensors"
    vae_model: str = "ema_vae_fp16.safetensors"
    attention_mode: str = "sdpa"
    device: str = "cuda:0"
    # These are intentionally separate. In the supplied ComfyUI workflow:
    # DiT offload=none, VAE offload=none, intermediate tensor offload=cpu.
    dit_offload_device: str = "none"
    vae_offload_device: str = "none"
    tensor_offload_device: str = "cpu"
    keep_model_cached: bool = False
    unload_forge_model_first: bool = True

    # Outer workflow tiling (replacement for TTP_Image_Tile_Batch / Assy)
    tile_size: int = 1024
    tile_padding: int = 64
    long_edge_tiles: int = 6

    # SeedVR2 node parameters
    color_correction: str = "lab"
    batch_size: int = 1
    blocks_to_swap: int = 0
    swap_io_components: bool = False
    uniform_batch_size: bool = False
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    seed: int = 0
    max_resolution: int = 0
    encode_tiled: bool = False
    encode_tile_size: int = 1024
    encode_tile_overlap: int = 128
    decode_tiled: bool = False
    decode_tile_size: int = 1024
    decode_tile_overlap: int = 128
    target_free_vram_gb: int = 18
    debug: bool = False

    def target_long_edge(self) -> int:
        # Mirrors the supplied ComfyUI graph: a*b-64, where a=tile count and b=tile size.
        return max(64, int(self.long_edge_tiles) * int(self.tile_size) - int(self.tile_padding))


def extension_root() -> Path:
    return Path(__file__).resolve().parents[1]


def forge_root() -> Path:
    # .../forge/extensions/sd-webui-seedvr2 -> .../forge
    return extension_root().parents[1]


def vendor_seedvr2_root() -> Path:
    return extension_root() / "vendor" / "seedvr2"


def models_root() -> Path:
    # Upstream uses models/SEEDVR2. Keep compatibility with the mixed-case
    # directory used by the first scaffold; on Windows these resolve equally.
    canonical = forge_root() / "models" / "SEEDVR2"
    legacy = forge_root() / "models" / "SeedVR2"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def list_model_files() -> List[str]:
    root = models_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".safetensors", ".gguf"}
    )


def require_model(filename: str) -> Path:
    path = models_root() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"SeedVR2 model not found: {path}\n"
            f"Put SeedVR2 model files in: {models_root()}"
        )
    return path
