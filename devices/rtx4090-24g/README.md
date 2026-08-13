# NVIDIA RTX 4090 24GB Backend

Target:

- NVIDIA GeForce RTX 4090 24GB
- Ada / `sm_89`
- direct CUDA compilation with `nvcc`
- BF16 model storage
- FP32 accumulation
- explicit CUDA kernels for model and codec operators
- no cuBLAS, cuBLASLt, CUTLASS, or cuDNN operator path

Supported model families:

- 0.6B Base
- 0.6B CustomVoice
- 1.7B Base
- 1.7B CustomVoice
- 1.7B VoiceDesign

Resident execution uses 80 SMs for generation and 48 SMs for codec decoding.
Base requests can reuse a bounded reference-audio speaker embedding cache inside
a Resident process.

Streaming contract:

- first audio packet: 8 codec frames / 640 ms of audio
- steady packet: 16 codec frames / 1280 ms of audio
- `--max-new-tokens` is required
- request range: 1..8192
