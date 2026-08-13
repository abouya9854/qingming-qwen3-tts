#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${ROOT_DIR}/models"

mkdir -p "${MODELS_DIR}"

declare -A MODELS=(
  ["tokenizer"]="Qwen3-TTS-Tokenizer-12Hz"
  ["0.6b-base"]="Qwen3-TTS-12Hz-0.6B-Base"
  ["0.6b-customvoice"]="Qwen3-TTS-12Hz-0.6B-CustomVoice"
  ["1.7b-base"]="Qwen3-TTS-12Hz-1.7B-Base"
  ["1.7b-customvoice"]="Qwen3-TTS-12Hz-1.7B-CustomVoice"
  ["1.7b-voicedesign"]="Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)

download_one() {
  local name="$1"
  local dst="${MODELS_DIR}/${name}"

  if command -v hf >/dev/null 2>&1; then
    hf download "Qwen/${name}" --local-dir "${dst}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "Qwen/${name}" --local-dir "${dst}"
  else
    echo "Hugging Face CLI not found. Install it with: pip install -U huggingface_hub" >&2
    exit 1
  fi
}

target="${1:-all}"

if [[ "${target}" == "all" ]]; then
  for key in tokenizer 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice 1.7b-voicedesign; do
    download_one "${MODELS[$key]}"
  done
elif [[ -n "${MODELS[$target]:-}" ]]; then
  download_one "${MODELS[$target]}"
else
  echo "Unknown model target: ${target}" >&2
  exit 1
fi
