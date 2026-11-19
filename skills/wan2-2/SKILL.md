---
name: wan2-2
description: Generate videos locally with Wan2.2 open video models, including text-to-video (T2V-A14B), image-to-video (I2V-A14B), the 5B text-image-to-video model that runs on a 24 GB GPU (TI2V-5B), audio-driven speech-to-video (S2V-14B) and character animation or replacement (Animate-14B). Use when the user mentions Wan or Wan2.2, wants to make a video from a prompt, an image or audio, animate or replace a character from a reference video, pick a Wan model for their GPU, download Wan weights, or fix out-of-memory errors in generate.py.
---

# Wan2.2

Inference code for the Wan2.2 video models. Everything runs through `generate.py` in this repository.

## Pick the model for the GPU

| Task | Model | `--task` | Sizes | Minimum VRAM (single GPU) |
|---|---|---|---|---|
| Text → video | T2V-A14B | `t2v-A14B` | `1280*720`, 480P | 80 GB |
| Image → video | I2V-A14B | `i2v-A14B` | `1280*720`, 480P | 80 GB |
| Text and/or image → video, 720P at 24 fps | **TI2V-5B** | `ti2v-5B` | `1280*704` | **24 GB** (RTX 4090) |
| Speech/audio → video | S2V-14B | `s2v-14B` | `1024*704`, 480P | 80 GB |
| Animate or replace a character | Animate-14B | `animate-14B` | from preprocessing | 80 GB class |

On a consumer GPU (24 GB or less), recommend **TI2V-5B**. Don't promise 14B results on small GPUs. Smaller cards may need community tools such as DiffSynth-Studio (layer-by-layer offload, FP8) or ComfyUI.

## Install

```bash
pip install -r requirements.txt          # needs torch >= 2.4.0
pip install -r requirements_s2v.txt      # only for speech-to-video with CosyVoice TTS
```

If `flash_attn` fails to install, install the other packages first and `flash_attn` last.

## Download weights

Weights are many GB. Confirm the model and the disk space with the user before downloading.

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

Other repos: `Wan-AI/Wan2.2-T2V-A14B`, `Wan-AI/Wan2.2-I2V-A14B`, `Wan-AI/Wan2.2-S2V-14B`, `Wan-AI/Wan2.2-Animate-14B`. ModelScope works too: `pip install modelscope`, then `modelscope download Wan-AI/<model> --local_dir ./<model>`.

## Generate

**TI2V-5B on a 24 GB GPU** (add `--image file.jpg` for image-to-video):

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu --prompt "..."
```

**Text-to-video, 14B:**

```bash
python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B \
  --offload_model True --convert_model_dtype --prompt "..."
```

**Image-to-video, 14B:** `--task i2v-A14B --ckpt_dir ./Wan2.2-I2V-A14B --image input.jpg`. The prompt can be empty when using prompt extension.

**Speech-to-video:**

```bash
python generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ \
  --offload_model True --convert_model_dtype --prompt "..." --image person.jpg --audio talk.wav
```

The length follows the audio unless `--num_clip` is set. Add `--pose_video pose.mp4` to follow a pose video. `--enable_tts --tts_prompt_audio ... --tts_prompt_text ... --tts_text ...` makes the audio with CosyVoice.

**Several GPUs:** `torchrun --nproc_per_node=8 generate.py ... --dit_fsdp --t5_fsdp --ulysses_size 8`, without the offload flags.

## Wan-Animate

It takes a video and a character image. **Animation** mode makes the character copy the video's motion. **Replacement** mode puts the character into the video in place of its person.

1. Preprocess:
   ```bash
   python ./wan/modules/animate/preprocess/preprocess_data.py \
     --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
     --video_path video.mp4 --refer_path image.jpeg --save_path ./process_results \
     --resolution_area 1280 720 --retarget_flag --use_flux
   ```
   For replacement, use `--iterations 3 --k 7 --w_len 1 --h_len 1 --replace_flag` instead of `--retarget_flag --use_flux`.
2. Generate:
   ```bash
   python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ \
     --src_root_path ./process_results/ --refert_num 1
   ```
   For replacement, add `--replace_flag`, and check `wan/modules/animate/preprocess/UserGuider.md` for the options it needs.

## Prompts and options

- Describe the subject, action, setting, camera and style in one or two detailed sentences, like the README's examples.
- **Prompt extension** rewrites short prompts: `--use_prompt_extend --prompt_extend_method dashscope` (needs `DASH_API_KEY`, which sends the prompt to Alibaba's cloud) or `local_qwen` (local, more VRAM). Ask before using the cloud option, and never print the key.
- **Out of memory:** add `--offload_model True --convert_model_dtype --t5_cpu`, lower the size, or move to TI2V-5B.
- **Diffusers:** T2V, I2V, TI2V and Animate also have Diffusers weights (`Wan-AI/<model>-Diffusers`), if the user prefers a Python pipeline.

Report the output file path when a run finishes, and never claim a video was made without seeing it.
