# AMD RX 7900 XTX 24GB Backend

Target:

- AMD Radeon RX 7900 XTX 24GB
- RDNA 3 / `gfx1100`
- HIP/ROCm direct compilation
- BF16 model storage
- FP32 accumulation
- Explicit device kernels for model and codec operators

Supported model families:

- 0.6B Base
- 0.6B CustomVoice
- 1.7B Base
- 1.7B CustomVoice
- 1.7B VoiceDesign

Resident execution uses 28 WGPs for generation and 20 WGPs for codec decoding.

Streaming contract:

- first audio packet: 8 codec frames / 640 ms of audio
- steady packet: 16 codec frames / 1280 ms of audio
- `--max-new-tokens` is required
- request range: 1..8192
