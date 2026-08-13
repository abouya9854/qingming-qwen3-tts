#pragma once


//   NVIDIA GeForce RTX 4090 24GB / AD102
//   CUDA target: sm_89
//
// Hardware contract:
//   SMs:                       128
//   warp size:                 32
//   resident threads / SM:     1536
//   shared memory / SM:        100 KiB
//   shared memory / block max: 99 KiB
//
// Kernel policy:
//   direct CUDA kernels only
//   no cuBLAS / cuBLASLt / CUTLASS / cuDNN dependency
//   BF16 model storage with FP32 accumulation
//   warp32 reductions and explicit shared-memory staging
//
// Initial execution family:
//   qwen3_tts_0_6b.cu
//     Qwen3-TTS 12Hz 0.6B Base / CustomVoice
//
// Streaming policy:
//   first audio chunk: 8 codec frames
//   steady chunk:      16 codec frames
//
// Resident scheduler:
//   CUDA >= 13.1: Green Context split, generation 80 SM / decoder 48 SM
//   older CUDA:    non-blocking priority streams
