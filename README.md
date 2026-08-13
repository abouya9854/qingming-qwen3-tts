# qingming-qwen3-tts — Device-Native Qwen3-TTS Inference on AMD RX 7900 XTX / NVIDIA RTX 4090

`qingming-qwen3-tts` is a device-native inference implementation for the Qwen3-TTS 12Hz model family.

The project targets concrete 24GB desktop GPUs and builds the execution path around those devices instead of routing model operators through a general-purpose deep-learning runtime.

The stable runtime contract is documented in [`契约.md`](./契约.md).

---

## At a Glance

| Item | AMD RX 7900 XTX 24GB | NVIDIA RTX 4090 24GB |
|---|---|---|
| Architecture target | `gfx1100` | `sm_89` |
| Toolchain | HIP / ROCm | CUDA / nvcc |
| Model families | 0.6B / 1.7B | 0.6B / 1.7B |
| Base x-vector | yes | yes |
| CustomVoice | yes | yes |
| CustomVoice + instruct | 1.7B | 1.7B |
| VoiceDesign | 1.7B | 1.7B |
| Lifecycle | Once / Resident | Once / Resident |
| First audio packet | 8 codec frames | 8 codec frames |
| Steady packet | 16 codec frames | 16 codec frames |
| `--max-new-tokens` | required, 1..8192 | required, 1..8192 |
| Model storage | BF16 | BF16 |
| Primary accumulation | FP32 | FP32 |

### Resident Realtime at a Glance

Higher is better.

| Family / Task | RX 7900 XTX | RTX 4090 |
|---|---:|---:|
| 0.6B Base | **2.99×** | **7.60×** |
| 0.6B CustomVoice | **3.47×** | **8.64×** |
| 1.7B Base | **2.55×** | **5.95×** |
| 1.7B CustomVoice + instruct | **2.55×** | **6.18×** |
| 1.7B VoiceDesign | **2.51×** | **6.18×** |

All values above come from the project benchmark harness with the same streaming contract:

```text
1 codec frame = 80 ms audio
first audio packet = 8 frames = 640 ms audio
steady packet = 16 frames = 1280 ms audio
benchmark = 10 Once requests + 10 Resident requests
seed = 1234
max-new-tokens = 512
```

---

## 1. Positioning

The project focuses on:

- AMD Radeon RX 7900 XTX 24GB (`gfx1100`)
- NVIDIA GeForce RTX 4090 24GB (`sm_89`)
- explicit low-level GPU kernels
- BF16 model storage with FP32 accumulation
- streaming speech generation
- Once and Resident lifecycles
- deterministic codec trajectories and WAV output across tested lifecycles
- explicit memory-capacity selection up to `--max-new-tokens 8192`

This is not a portable framework backend. Each device directory owns its execution policy, kernel topology, memory layout, concurrency strategy, and streaming behavior.

### Execution Path

```text
Text / speaker / instruction / reference audio
                    │
                    ▼
              Qwen3-TTS Talker
                    │
                    ▼
                 Predictor
                    │
                    ▼
             12 Hz codec frames
                    │
          ┌─────────┴─────────┐
          │                   │
   first 8 frames       steady 16 frames
          │                   │
          └─────────┬─────────┘
                    ▼
              Codec decoder
                    │
                    ▼
                   WAV
```

Resident mode keeps model state and persistent device allocations alive across requests.

Device-local execution policy:

| Backend | Resident generation | Resident decoder |
|---|---:|---:|
| RX 7900 XTX | 28 WGP | 20 WGP |
| RTX 4090 | 80 SM | 48 SM |

The NVIDIA backend executes model and codec operators with explicit CUDA kernels and does not use cuBLAS, cuBLASLt, CUTLASS, or cuDNN as its model-operator execution path.

The AMD backend executes the supported paths with explicit HIP device kernels and does not require a high-level framework operator runtime.

---

## 2. Thanks to the Official Qwen3-TTS Project

This project exists because of the work of the Qwen team on Qwen3-TTS.

Official resources:

- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Qwen3-TTS Technical Report: https://arxiv.org/abs/2601.15621
- Qwen organization on Hugging Face: https://huggingface.co/Qwen

The model architecture, checkpoints, tokenizer, speaker definitions, and model semantics belong to the upstream Qwen3-TTS project. This repository focuses on device-native inference execution for the supported GPUs.

---

## 3. Results

### Test Environment and Definitions

Unless a row says otherwise:

- benchmark mode: 10 Once requests + 10 Resident requests
- seed: `1234`
- `--max-new-tokens 512`
- text mode: `streaming`
- first audio packet: 8 codec frames = 640 ms of generated audio
- steady streaming packet: 16 codec frames = 1280 ms of generated audio
- 1 codec frame = 80 ms of audio
- TTFT: time to the first generated codec frame
- TTFA: time until the first 8-frame audio packet is available
- E2E: end-to-end request latency reported by the runtime
- Resident RTF: `Resident request wall mean / generated audio duration`
- Resident Realtime: `generated audio duration / Resident request wall mean`
- lower RTF is better
- higher Realtime is better

Formula:

```text
audio duration = codec frames × 80 ms
Resident RTF = request wall mean / audio duration
Resident Realtime = audio duration / request wall mean
```

### Correctness Gate

Every benchmark row shown below passed:

| Check | Result |
|---|---|
| Once success | 10/10 |
| Resident success | 10/10 |
| Once EOS | PASS |
| Resident EOS | PASS |
| Once frame consistency | PASS |
| Resident frame consistency | PASS |
| Cross-lifecycle frame match | PASS |
| Once codec trajectory consistency | PASS |
| Resident codec trajectory consistency | PASS |
| Once vs Resident codec trajectory | exact |
| Once WAV byte consistency | PASS |
| Resident WAV byte consistency | PASS |
| Once vs Resident WAV bytes | exact |
| Trajectory accuracy | PASS |

These figures are device-local measurements from this project. They are not normalized against another framework server, batching policy, concurrency level, or a different first-packet definition.

### 3.1 AMD RX 7900 XTX 24GB

Test target:

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 7900 XTX |
| VRAM | 24 GB |
| ISA target | `gfx1100` |
| Toolchain | HIP / ROCm |
| CMake ROCm path | `/opt/rocm-7.2.4` |
| Resident allocation | 28 WGP generation / 20 WGP decoder |
| First / steady packet | 8 / 16 codec frames |
| Maximum request | 8192 new tokens |

Latest benchmark records:

| Family / Task | Frames | Once TTFT mean | Once E2E mean | Resident TTFT p50 | Resident TTFA p50 | Resident E2E mean | Resident RTF | Resident Realtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B Base | 63 | 242.019 ms | 1890.278 ms | 138.243 ms | 443.180 ms | 1683.730 ms | 0.334 | **2.99×** |
| 0.6B CustomVoice | 123 | 208.748 ms | 3492.823 ms | 42.962 ms | 348.399 ms | 2834.079 ms | 0.288 | **3.47×** |
| 1.7B Base | 68 | 363.875 ms | 2566.653 ms | 182.475 ms | 526.482 ms | 2132.219 ms | 0.392 | **2.55×** |
| 1.7B CustomVoice + instruct | 76 | 401.848 ms | 2698.227 ms | 175.649 ms | 519.821 ms | 2387.751 ms | 0.393 | **2.55×** |
| 1.7B VoiceDesign | 73 | 425.902 ms | 2706.109 ms | 192.613 ms | 536.374 ms | 2325.727 ms | 0.398 | **2.51×** |

Resident request-wall means used for the RTF / Realtime calculation:

```text
0.6B Base                    1683.833 ms
0.6B CustomVoice             2834.173 ms
1.7B Base                    2132.316 ms
1.7B CustomVoice + instruct  2387.849 ms
1.7B VoiceDesign             2325.825 ms
```

Validated model/task coverage:

| Family | Task | Once | Resident | Codec trajectory | WAV bytes |
|---|---|---:|---:|---:|---:|
| 0.6B | Base x-vector | PASS | PASS | exact | exact |
| 0.6B | CustomVoice | PASS | PASS | exact | exact |
| 1.7B | Base x-vector | PASS | PASS | exact | exact |
| 1.7B | CustomVoice | PASS | PASS | exact | exact |
| 1.7B | CustomVoice + instruct | PASS | PASS | exact | exact |
| 1.7B | VoiceDesign | PASS | PASS | exact | exact |

Long-capacity validation:

| Family / Task | Lifecycle | Capacity | Frames | Audio | EOS |
|---|---|---:|---:|---:|---:|
| 1.7B Base | Once | 8192 | 4304 | 344.320 s | natural EOS |

### 3.2 NVIDIA RTX 4090 24GB

Test target:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| VRAM | 24 GB |
| ISA target | `sm_89` |
| CUDA toolkit | CUDA 13.2 |
| nvcc | V13.2.51 |
| Resident allocation | 80 SM generation / 48 SM decoder |
| First / steady packet | 8 / 16 codec frames |
| Maximum request | 8192 new tokens |

Latest benchmark records:

| Family / Task | Frames | Once TTFT mean | Once E2E mean | Resident TTFT p50 | Resident TTFA p50 | Resident E2E mean | Resident RTF | Resident Realtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B Base | 53 | 565.659 ms | 1256.458 ms | 12.470 ms | 128.774 ms | 557.299 ms | 0.132 | **7.60×** |
| 0.6B CustomVoice | 102 | 509.865 ms | 1805.442 ms | 12.754 ms | 129.019 ms | 944.244 ms | 0.116 | **8.64×** |
| 1.7B Base | 62 | 835.101 ms | 1940.043 ms | 18.466 ms | 156.502 ms | 833.975 ms | 0.168 | **5.95×** |
| 1.7B CustomVoice + instruct | 69 | 876.840 ms | 2070.667 ms | 27.498 ms | 165.907 ms | 892.676 ms | 0.162 | **6.18×** |
| 1.7B VoiceDesign | 69 | 847.715 ms | 2014.684 ms | 30.082 ms | 168.245 ms | 892.366 ms | 0.162 | **6.18×** |

Resident request-wall means used for the RTF / Realtime calculation:

```text
0.6B Base                     557.724 ms
0.6B CustomVoice              944.662 ms
1.7B Base                     834.141 ms
1.7B CustomVoice + instruct   892.864 ms
1.7B VoiceDesign              892.560 ms
```

The 0.6B Base Resident path can reuse a bounded reference-audio speaker embedding cache inside the Resident process.

---

## 4. Model Download

Official Hugging Face repositories:

| Local directory | Official repository |
|---|---|
| `Qwen3-TTS-Tokenizer-12Hz` | https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz |
| `Qwen3-TTS-12Hz-0.6B-Base` | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice |
| `Qwen3-TTS-12Hz-1.7B-Base` | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign |

Download all supported checkpoints:

```bash
pip install -U huggingface_hub
./download.sh all
```

Or download one model:

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

Supported `download.sh` targets:

```text
tokenizer
0.6b-base
0.6b-customvoice
1.7b-base
1.7b-customvoice
1.7b-voicedesign
```

---

## 5. Scope

Stable runtime coverage:

| Family | Base voice clone | CustomVoice | CustomVoice + instruct | VoiceDesign |
|---|---:|---:|---:|---:|
| 0.6B | yes | yes | outside stable contract | no |
| 1.7B | yes | yes | yes | yes |

Additional constraints:

- text mode: `streaming`
- Base requires `--ref-audio`
- CustomVoice requires `--speaker`
- VoiceDesign requires `--instruct`
- use an explicit language for the stable Resident contract
- `--max-new-tokens` is mandatory
- supported request range: 1..8192
- internal capacity classes: 256 / 512 / 1024 / 2048 / 4096 / 8192
- Resident startup allocates its capacity before serving requests
- each Resident request must not exceed the Resident process capacity
- output is WAV audio
- stable model storage is BF16
- primary accumulation is FP32
- checkpoints outside the listed official model families are outside the stable contract
- multi-GPU execution is outside the stable contract
- server-side dynamic batching is outside the stable contract
- training and fine-tuning are outside the stable contract

### Repository Layout

```text
qingming-qwen3-tts/
├── CMakeLists.txt
├── README.md
├── 契约.md
├── LICENSE
├── download.sh
├── main.cpp
├── models/
│   ├── Qwen3-TTS-Tokenizer-12Hz/
│   ├── Qwen3-TTS-12Hz-0.6B-Base/
│   ├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
│   ├── Qwen3-TTS-12Hz-1.7B-Base/
│   ├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   └── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
└── devices/
    ├── rx7900xtx-24g/
    │   ├── benchmark.cpp
    │   ├── device.h
    │   ├── qwen3_tts_0_6b.cpp
    │   └── qwen3_tts_1_7b.cpp
    └── rtx4090-24g/
        ├── benchmark.cpp
        ├── device.h
        ├── main.cu
        ├── qwen3_tts_0_6b.cu
        └── qwen3_tts_1_7b.cu
```

Model files are downloaded separately and are not bundled in the source archive.

---

## 6. Build and Run

### 6.1 AMD RX 7900 XTX 24GB

Build 0.6B:

```bash
rm -rf build/rx7900xtx-24g-0.6b

cmake \
  -S . \
  -B build/rx7900xtx-24g-0.6b \
  -DQINGMING_DEVICE=rx7900xtx-24g \
  -DQINGMING_MODEL_FAMILY=0.6b \
  -DROCM_PATH=/opt/rocm-7.2.4

cmake --build build/rx7900xtx-24g-0.6b -j
```

Build 1.7B:

```bash
rm -rf build/rx7900xtx-24g-1.7b

cmake \
  -S . \
  -B build/rx7900xtx-24g-1.7b \
  -DQINGMING_DEVICE=rx7900xtx-24g \
  -DQINGMING_MODEL_FAMILY=1.7b \
  -DROCM_PATH=/opt/rocm-7.2.4

cmake --build build/rx7900xtx-24g-1.7b -j
```

Example Base request:

```bash
./build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b \
  --lifecycle once \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-Base \
  --task base-xvector \
  --text-mode streaming \
  --ref-audio ./clone.wav \
  --language English \
  --text "Hello from qingming-qwen3-tts." \
  --output ./rx7900xtx-base.wav \
  --seed 1234 \
  --max-new-tokens 512
```

Example benchmark:

```bash
./build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_benchmark \
  --family 1.7b \
  --task base-xvector \
  --text-mode streaming \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio ./clone.wav \
  --language English \
  --text "Hello from the RX 7900 XTX 1.7B Base benchmark." \
  --seed 1234 \
  --max-new-tokens 512 \
  --output-dir ./benchmark-rx7900xtx-24g-1.7b-base \
  --report-json ./benchmark-rx7900xtx-24g-1.7b-base.json \
  --report-txt ./benchmark-rx7900xtx-24g-1.7b-base.txt
```

### 6.2 NVIDIA RTX 4090 24GB

Build 0.6B:

```bash
rm -rf build/rtx4090-24g-0.6b

cmake \
  -S . \
  -B build/rtx4090-24g-0.6b \
  -DQINGMING_DEVICE=rtx4090-24g \
  -DQINGMING_MODEL_FAMILY=0.6b \
  -DCUDA_PATH=/usr/local/cuda

cmake --build build/rtx4090-24g-0.6b -j
```

Build 1.7B:

```bash
rm -rf build/rtx4090-24g-1.7b

cmake \
  -S . \
  -B build/rtx4090-24g-1.7b \
  -DQINGMING_DEVICE=rtx4090-24g \
  -DQINGMING_MODEL_FAMILY=1.7b \
  -DCUDA_PATH=/usr/local/cuda

cmake --build build/rtx4090-24g-1.7b -j
```

Example Base request:

```bash
./build/rtx4090-24g-1.7b/qingming-qwen3-tts_rtx4090-24g_1.7b \
  --lifecycle once \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-Base \
  --task base-xvector \
  --text-mode streaming \
  --ref-audio ./clone.wav \
  --language English \
  --text "Hello from qingming-qwen3-tts." \
  --output ./rtx4090-base.wav \
  --seed 1234 \
  --max-new-tokens 512
```

Example CustomVoice + instruct:

```bash
./build/rtx4090-24g-1.7b/qingming-qwen3-tts_rtx4090-24g_1.7b \
  --lifecycle once \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --task custom-voice \
  --text-mode streaming \
  --speaker Ryan \
  --language English \
  --instruct "Speak in a calm and confident tone." \
  --text "Hello from qingming-qwen3-tts." \
  --output ./rtx4090-custom.wav \
  --seed 1234 \
  --max-new-tokens 512
```

Example VoiceDesign:

```bash
./build/rtx4090-24g-1.7b/qingming-qwen3-tts_rtx4090-24g_1.7b \
  --lifecycle once \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --task voice-design \
  --text-mode streaming \
  --language English \
  --instruct "A warm, mature male voice with a calm and confident delivery." \
  --text "Hello from qingming-qwen3-tts." \
  --output ./rtx4090-design.wav \
  --seed 1234 \
  --max-new-tokens 512
```

For a Resident process, use:

```text
--lifecycle resident
```

and follow the JSON-line request contract documented in [`契约.md`](./契约.md).

---

## 7. License

This repository is distributed under the Apache License 2.0. See [`LICENSE`](./LICENSE).

Qwen3-TTS, its official source code, tokenizer, and model checkpoints are provided by the Qwen team. Their upstream copyright notices, licenses, model cards, and usage terms remain applicable.
