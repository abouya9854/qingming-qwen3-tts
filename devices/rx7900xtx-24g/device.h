#pragma once


//   AMD Radeon RX 7900 XTX 24GB
//   HIP target: gfx1100
//
// Runtime topology:
//   physical RDNA3 CUs:                 96
//   HIP multiProcessorCount (WGP mode): 48
//   1 WGP:                              2 physical CUs
//
// Execution families:
//   qwen3_tts_1_7b.cpp
//     Qwen3-TTS 12Hz 1.7B
//     --task base-xvector  --text-mode streaming
//     --task custom-voice  --text-mode streaming
//     --task voice-design  --text-mode streaming
//     CustomVoice accepts optional --instruct.
//     VoiceDesign requires --instruct.
//
//   qwen3_tts_0_6b.cpp
//     Qwen3-TTS 12Hz 0.6B
//     --task base-xvector  --text-mode streaming
//     --task custom-voice  --text-mode streaming
//     CustomVoice requires --speaker and does not accept --instruct.
//
// Resident execution partition:
//   generation 28 WGP
//   decoder    20 WGP
//
// benchmark.cpp runs independent Once processes and sequential Resident
// requests against the selected family binary.
