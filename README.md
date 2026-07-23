# sd-webui-seedvr2-forge

A Forge-Neo post-processing extension that ports a ComfyUI SeedVR2 tiled-image workflow into WebUI Forge.

## Preset reproduced from the supplied workflow

- DiT: `seedvr2_ema_3b_fp16.safetensors`
- VAE: `ema_vae_fp16.safetensors`
- inference device: `cuda:0`
- DiT offload: `none`
- VAE offload: `none`
- intermediate tensor offload: `cpu`
- attention: `sdpa`
- SeedVR2 frame batch: `1`
- LAB color correction
- VAE encode/decode tiling: off
- input noise / latent noise: `0`
- BlockSwap: `0`
- outer tile: `1024 x 1024`
- long-edge tile count: `6`
- pre-resize target: `6 * 1024 - 64 = 6080px` long edge
- TTP-compatible adaptive tile positions and assembly blending, with assembly padding `64`

## Installation (Windows / Forge-Neo)

1. Extract this folder as:

   `sd-webui-forge-neo/extensions/sd-webui-seedvr2/`

2. Restart Forge-Neo. Forge's extension installer should run `install.py` automatically.
   The installer deliberately **does not install/upgrade torch, torchvision or numpy**.

3. The installer downloads the official SeedVR2 `v2.5.23` source into:

   `extensions/sd-webui-seedvr2/vendor/seedvr2/`

   If GitHub is inaccessible, manually copy the contents of the official
   `numz/ComfyUI-SeedVR2_VideoUpscaler` v2.5.23 repository into that folder.

4. Copy your existing model files to:

   `sd-webui-forge-neo/models/SEEDVR2/`

   Required for the supplied preset:

   - `seedvr2_ema_3b_fp16.safetensors`
   - `ema_vae_fp16.safetensors`

5. Restart Forge-Neo and open the post-processing / Extras interface. Enable the
   **SeedVR2 Upscaler** accordion.

## RTX 3090 preset

Use the defaults first:

- 3B FP16 DiT
- FP16 VAE
- SDPA
- batch 1
- DiT/VAE offload `none`
- tensor offload `cpu`
- BlockSwap `0`
- `Unload Forge model before SeedVR2 = on`
- `Keep SeedVR2 models cached = off`

If the encoding or decoding phase runs out of VRAM, enable VAE tiled encode/decode.
If the DiT phase runs out of VRAM, set DiT offload to `cpu` before enabling BlockSwap.

## Important host dependency policy

Forge-Neo owns its PyTorch stack and pins several Python libraries. Upstream SeedVR2
has its own requirements. This extension intentionally reuses the Forge environment
instead of blindly installing SeedVR2's entire `requirements.txt`, because replacing
Forge's `torch`/`torchvision`/`numpy` can break the WebUI.

In particular, current Forge-Neo pins an OmegaConf version older than the minimum
listed by upstream SeedVR2. The installer preserves the host version instead of
silently upgrading a Forge-pinned package. If this becomes a real runtime incompatibility
on your checkout, solve it in the adapter rather than globally upgrading Forge first.

## Offline smoke test

From the extension directory:

`python tests/smoke_test.py`

This validates the 6080 calculation and TTP-equivalent split/merge logic. It does not
load CUDA models.

## Known limitation of this build

The adapter has been matched against the official SeedVR2 v2.5.23 core function
signatures and passes local syntax/tiling smoke tests. End-to-end GPU inference cannot
be executed in the build environment because the multi-GB SeedVR2 weights and an RTX
GPU are not present here. The first real Forge run on your machine is therefore the
final integration test for model-loading/runtime-library compatibility.
