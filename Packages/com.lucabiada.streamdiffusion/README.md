# StreamDiffusionUnity

Interactive realtime image generation ([StreamDiffusion](https://github.com/LucaBiada/StreamDiffusion)) controlled from Unity 6 (URP).
A runtime component and UI drive a headless Python server (WebSocket control, shared-memory video, optional NDI output).
It works in the Editor and in an exported Windows build.

## Requirements

- Windows 10/11, Direct3D 11 (set as the only graphics API for Windows standalone).
- NVIDIA RTX GPU with a recent driver (developed on an RTX 5090, CUDA 12.8).
- Unity 6000.6 with the Universal Render Pipeline.
- About 10 GB of disk for the Python environment, plus TensorRT engines (a few GB per model/resolution).
- Git (or let the installer download MinGit) and Python 3.11 (or let it download the embeddable build).

## Install

1. Create or open a Unity 6000.6 URP project.
2. Package Manager > + > *Add package from git URL*:

   ```
   https://github.com/LucaBiada/StreamDiffusionUnity.git?path=/Packages/com.lucabiada.streamdiffusion#v0.1.0
   ```

3. Package Manager > StreamDiffusionUnity > Samples > *Import* the sample scene, and open it.
4. Press Play. In the panel open **Environment** and press **Install environment**. It clones
   `LucaBiada/StreamDiffusion` and the installer from your forks, creates the venv (PyTorch, TensorRT, ...)
   and verifies it (about 25 minutes, one time). Give the folder a path without spaces
   (default: `%LOCALAPPDATA%\StreamDiffusionUnity\diffusion`).
5. **Connect & Start**. The first load of a model/resolution builds TensorRT engines (minutes of full GPU load;
   the panel asks for confirmation first, and can write a build script to run with Unity closed).

The same repository is also a ready-to-open Unity project: clone it and open the folder with Unity 6000.6.

## Contents

| Folder | What |
|---|---|
| `Runtime/` | controller, installer, server launcher, transport (shared memory), UI panel |
| `Editor/` | inspector and the post-build step that copies the server next to the exe |
| `Server~/` | the Python server (`run_server.py`, `server/`, `tools/`, tests). Copied next to the exe on build |
| `ThirdParty/DesignSystem/` | the UI Toolkit design system the panel uses (MIT, see `THIRD_PARTY_NOTICES.md`) |
| `Samples~/` | sample scene, prefab, render textures, depth-map camera |

## Command line (player)

`-sd-basefolder <dir> -sd-install -sd-allow-downloads -sd-autostart -sd-host -sd-port -sd-token -sd-no-gpu-fix`

## Notes

- Building engines can push the GPU hard; the installer offers to raise the Windows display-driver timeout
  (`TdrDelay`) to 60 s, which needs administrator permission and a restart.
- Everything the installer fetches comes from the `LucaBiada` forks (StreamDiffusion, installer, diffusers,
  Diffusers_IPAdapter, cuda-link wheel, insightface wheel and the `buffalo_l` face pack), plus PyPI, the PyTorch/NVIDIA
  indexes and Hugging Face for model weights.
