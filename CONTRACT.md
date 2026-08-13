# Runtime Contract

This document defines the stable runtime contract for `qingming-qwen3-tts`.

## Supported Devices

| Backend | Target |
|---|---|
| `rx7900xtx-24g` | AMD Radeon RX 7900 XTX 24GB / `gfx1100` |
| `rtx4090-24g` | NVIDIA GeForce RTX 4090 24GB / `sm_89` |

Each build is device-specific.

## Supported Models

| Family | Base | CustomVoice | CustomVoice + instruct | VoiceDesign |
|---|---:|---:|---:|---:|
| 0.6B | yes | yes | no | no |
| 1.7B | yes | yes | yes | yes |

Supported official model directories:

```text
Qwen3-TTS-Tokenizer-12Hz
Qwen3-TTS-12Hz-0.6B-Base
Qwen3-TTS-12Hz-0.6B-CustomVoice
Qwen3-TTS-12Hz-1.7B-Base
Qwen3-TTS-12Hz-1.7B-CustomVoice
Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

## Precision

- model storage: BF16
- primary accumulation: FP32

## Lifecycle

Supported lifecycles:

```text
once
resident
```

`once` runs one request and exits.

`resident` keeps model state and device allocations alive across requests.

Resident execution starts directly and does not require a prior Once request.

## Request Fields

Common required fields:

```text
--lifecycle once|resident
--model-dir <path>
--task <task>
--text-mode streaming
--language <language>
--text <text>
--seed <integer>
--max-new-tokens <1..8192>
```

Task-specific fields:

Base:

```text
--task base-xvector
--ref-audio <wav>
```

CustomVoice:

```text
--task custom-voice
--speaker <speaker>
```

1.7B CustomVoice with instruction:

```text
--instruct <instruction>
```

VoiceDesign:

```text
--task voice-design
--instruct <voice-description>
```

## Token Capacity

`--max-new-tokens` is mandatory.

Supported range:

```text
1..8192
```

Internal capacity classes:

```text
256
512
1024
2048
4096
8192
```

A Resident request must not exceed the capacity selected when the Resident process starts.

Natural EOS may stop generation before the requested token limit.

## Streaming

One codec frame represents 80 ms of audio.

```text
first audio packet:  8 frames  = 640 ms audio
steady packet:      16 frames = 1280 ms audio
```

The final packet may contain fewer frames when EOS is reached.

## Resident Device Allocation

RX 7900 XTX:

```text
generation: 28 WGP
decoder:    20 WGP
```

RTX 4090:

```text
generation: 80 SM
decoder:    48 SM
```

## Determinism

For a fixed model, request, input asset, and seed, validated paths require:

- repeated Once requests produce the same codec trajectory
- repeated Resident requests produce the same codec trajectory
- Once and Resident produce the same codec trajectory
- repeated Once requests produce byte-identical WAV output
- repeated Resident requests produce byte-identical WAV output
- Once and Resident produce byte-identical WAV output
- emitted frame count matches EOS position

## Output

Successful execution emits machine-readable JSON containing request and timing information.

Stable fields include:

```text
status
device
family
lifecycle
task
text_mode
seed
max_new_tokens
max_new_tokens_capacity
frames
codec_fnv1a64
eos
eos_frame
audio_duration_s
predictor_p50_ms
predictor_p95_ms
ttft_ms
ttfa_ms
first_audio_chunk_frames
first_audio_chunk_duration_ms
streaming_chunk_frames
generation_ms
decoder_ms
e2e_ms
output
```

Additional backend-specific fields may be present.

## Compatibility Boundary

The stable contract does not cover:

- arbitrary AMD or NVIDIA GPUs
- unsupported checkpoint dimensions
- quantized model variants
- multi-GPU execution
- server-side dynamic batching
- training or fine-tuning
- byte-identical output after changing model assets, precision, sampling rules, or request parameters
