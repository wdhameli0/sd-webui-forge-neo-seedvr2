from __future__ import annotations

import random
import sys
from pathlib import Path

# Make extension-local packages importable even if Forge does not prepend the
# extension root to sys.path in a particular launch mode.
_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

import gradio as gr
from PIL import Image

from modules import scripts_postprocessing

from seedvr2_forge.config import SeedVR2Config, list_model_files, models_root
from seedvr2_forge.engine import SeedVR2ForgeEngine
from seedvr2_forge.image_utils import resize_to_long_edge
from seedvr2_forge.memory import unload_forge_models
from seedvr2_forge.tiling import merge_tiles, split_image


def _model_choices(default_name: str):
    found = list_model_files()
    if default_name not in found:
        found.insert(0, default_name)
    return found


class SeedVR2PostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "SeedVR2"
    order = 2000

    def ui(self):
        with gr.Accordion("SeedVR2 Upscaler", open=False):
            enabled = gr.Checkbox(label="Enable SeedVR2", value=False)
            gr.Markdown(
                f"Model folder: `{models_root()}`  \n"
                "Workflow preset: 6 long-edge tiles · 1024 tile · 64 assembly padding · LAB · SDPA"
            )

            with gr.Row():
                dit_model = gr.Dropdown(
                    label="DiT model",
                    choices=_model_choices("seedvr2_ema_3b_fp16.safetensors"),
                    value="seedvr2_ema_3b_fp16.safetensors",
                    allow_custom_value=True,
                )
                vae_model = gr.Dropdown(
                    label="VAE model",
                    choices=_model_choices("ema_vae_fp16.safetensors"),
                    value="ema_vae_fp16.safetensors",
                    allow_custom_value=True,
                )

            with gr.Row():
                attention_mode = gr.Dropdown(
                    label="Attention",
                    choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                    value="sdpa",
                )
                device = gr.Textbox(label="DiT / VAE inference device", value="cuda:0")

            with gr.Row():
                long_edge_tiles = gr.Slider(label="Long-edge tile count", minimum=1, maximum=12, step=1, value=6)
                tile_size = gr.Slider(label="Outer tile size", minimum=256, maximum=2048, step=64, value=1024)
                tile_padding = gr.Slider(label="Assembly blend padding", minimum=0, maximum=256, step=8, value=64)

            with gr.Row():
                color_correction = gr.Dropdown(
                    label="Color correction",
                    choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    value="lab",
                )
                # 4n+1 is a SeedVR2 requirement. For this tiled-image workflow, 1 is the
                # correct value because unrelated spatial tiles should not be treated as
                # temporally coherent video frames.
                batch_size = gr.Dropdown(
                    label="SeedVR2 frame batch (keep 1 for tiled images)",
                    choices=[1, 5, 9, 13],
                    value=1,
                )
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            with gr.Row():
                input_noise_scale = gr.Slider(label="Input noise scale", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                latent_noise_scale = gr.Slider(label="Latent noise scale", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

            with gr.Accordion("Memory / offload", open=False):
                gr.Markdown(
                    "The defaults reproduce your ComfyUI graph: **DiT offload = none**, "
                    "**VAE offload = none**, **intermediate tensor offload = cpu**."
                )
                with gr.Row():
                    dit_offload_device = gr.Dropdown(
                        label="DiT offload device",
                        choices=["none", "cpu"],
                        value="none",
                    )
                    vae_offload_device = gr.Dropdown(
                        label="VAE offload device",
                        choices=["none", "cpu"],
                        value="none",
                    )
                    tensor_offload_device = gr.Dropdown(
                        label="Intermediate tensor offload",
                        choices=["cpu", "none"],
                        value="cpu",
                    )

                with gr.Row():
                    blocks_to_swap = gr.Slider(label="Blocks to swap", minimum=0, maximum=36, step=1, value=0)
                    swap_io_components = gr.Checkbox(label="Swap I/O components", value=False)

                with gr.Row():
                    unload_forge_model_first = gr.Checkbox(
                        label="Unload Forge model before SeedVR2",
                        value=True,
                    )
                    keep_model_cached = gr.Checkbox(
                        label="Keep SeedVR2 models cached",
                        value=False,
                    )
                    debug = gr.Checkbox(label="Verbose SeedVR2 log", value=False)

                target_free_vram_gb = gr.Slider(
                    label="Request free VRAM before loading SeedVR2 (GB)",
                    minimum=8,
                    maximum=24,
                    step=1,
                    value=18,
                )

            with gr.Accordion("VAE tiling / low VRAM", open=False):
                with gr.Row():
                    encode_tiled = gr.Checkbox(label="VAE tiled encode", value=False)
                    encode_tile_size = gr.Slider(label="Encode tile size", minimum=256, maximum=2048, step=64, value=1024)
                    encode_tile_overlap = gr.Slider(label="Encode overlap", minimum=0, maximum=512, step=16, value=128)
                with gr.Row():
                    decode_tiled = gr.Checkbox(label="VAE tiled decode", value=False)
                    decode_tile_size = gr.Slider(label="Decode tile size", minimum=256, maximum=2048, step=64, value=1024)
                    decode_tile_overlap = gr.Slider(label="Decode overlap", minimum=0, maximum=512, step=16, value=128)

        return {
            "enabled": enabled,
            "dit_model": dit_model,
            "vae_model": vae_model,
            "attention_mode": attention_mode,
            "device": device,
            "dit_offload_device": dit_offload_device,
            "vae_offload_device": vae_offload_device,
            "tensor_offload_device": tensor_offload_device,
            "long_edge_tiles": long_edge_tiles,
            "tile_size": tile_size,
            "tile_padding": tile_padding,
            "color_correction": color_correction,
            "batch_size": batch_size,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "input_noise_scale": input_noise_scale,
            "latent_noise_scale": latent_noise_scale,
            "seed": seed,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "unload_forge_model_first": unload_forge_model_first,
            "keep_model_cached": keep_model_cached,
            "target_free_vram_gb": target_free_vram_gb,
            "debug": debug,
        }

    def process(self, pp, **kwargs):
        if not kwargs.get("enabled", False):
            return

        # Gradio sliders/numbers may arrive as floats even when semantically integral.
        for key in (
            "long_edge_tiles",
            "tile_size",
            "tile_padding",
            "batch_size",
            "blocks_to_swap",
            "target_free_vram_gb",
            "encode_tile_size",
            "encode_tile_overlap",
            "decode_tile_size",
            "decode_tile_overlap",
            "seed",
        ):
            kwargs[key] = int(kwargs[key])

        if kwargs["batch_size"] not in (1, 5, 9, 13):
            raise ValueError("SeedVR2 batch size must follow 4n+1: 1, 5, 9, 13, ...")
        if kwargs["seed"] < 0:
            kwargs["seed"] = random.randint(0, 2**31 - 1)

        cfg = SeedVR2Config(**kwargs)

        if cfg.unload_forge_model_first:
            print("[SeedVR2 Forge] " + unload_forge_models(cfg.target_free_vram_gb))

        source: Image.Image = pp.image.convert("RGB")
        target_long_edge = cfg.target_long_edge()
        resized = resize_to_long_edge(source, target_long_edge, method="lanczos")
        layout = split_image(resized, tile_size=cfg.tile_size, tile_padding=cfg.tile_padding)

        print(
            f"[SeedVR2 Forge] {source.size} -> {resized.size}; "
            f"grid={layout.grid_size}, tiles={len(layout.tiles)}, "
            f"tile={cfg.tile_size}, assembly_padding={cfg.tile_padding}"
        )

        engine = SeedVR2ForgeEngine(cfg)
        try:
            processed_tiles = engine.process_tiles([tile.image for tile in layout.tiles])
            merged = merge_tiles(layout, processed_tiles)
            pp.image = merged

            if hasattr(pp, "nametags") and isinstance(pp.nametags, list):
                pp.nametags.append("seedvr2")
            if hasattr(pp, "info") and isinstance(pp.info, dict):
                pp.info["SeedVR2"] = self._build_info(
                    cfg=cfg,
                    upstream_version=engine.upstream_version,
                    tile_count=len(layout.tiles),
                    grid_size=layout.grid_size,
                    original_size=source.size,
                    resized_size=resized.size,
                )
        finally:
            engine.cleanup()

    @staticmethod
    def _build_info(cfg, upstream_version, tile_count, grid_size, original_size, resized_size):
        return (
            f"v{upstream_version}; model={cfg.dit_model}; vae={cfg.vae_model}; "
            f"orig={original_size[0]}x{original_size[1]}; "
            f"output={resized_size[0]}x{resized_size[1]}; "
            f"grid={grid_size[0]}x{grid_size[1]}; tiles={tile_count}; "
            f"tile={cfg.tile_size}; assembly_padding={cfg.tile_padding}; "
            f"attention={cfg.attention_mode}; color={cfg.color_correction}; "
            f"batch={cfg.batch_size}; blocks_to_swap={cfg.blocks_to_swap}; seed={cfg.seed}"
        )
