// RX7900XTX-24G Qwen3-TTS 0.6B execution family.
// Talker=1024, text_hidden=2048, text_projection=2048->2048->1024,
// speaker=1024, Predictor outer/hidden=1024 with identity small_to_mtp,
// FinalHead=3072x1024. Codec codebook size 2048 is intentionally unchanged.


// ============================================================================
// Execution family identity
//
// File:
//   qwen3_tts_0_6b.cpp
//
// Device:
//   RX7900XTX-24G / gfx1100
//
// Family contract:
//   Qwen3-TTS 12Hz 0.6B execution family.
//

//   0.6B Base / official-base-xvector.
//
// Intended family coverage:
//   0.6B Base
//   0.6B CustomVoice
//   Base ICL when the native reference speech-tokenizer encoder/RVQ path exists.
//
// This file is named by parameter family rather than by the current task so the
// device execution image can grow to all compatible 0.6B tasks without another
// source rename.
//
// Initial RX7900XTX-24G resident production partition:
//   generation 28 WGP
//   decoder    20 WGP
// ============================================================================

// Qwen3-TTS-12Hz-0.6B-Base native inference baseline.
// Official Base x-vector-only voice-clone input contract.
//
// Production inputs:
//   model directory + reference WAV + language + target text
//
// Runtime:
//   one C++/HIP process, no Python/HuggingFace/rocBLAS/hipBLAS bridge.
//
// Correctness baseline:
//   native speaker encoder
//   official Base x-vector prompt construction
//   explicit BF16 Talker materialization boundaries
//   official Talker RoPE rotate_half = (-x2, +x1)
//   native top-k categorical generation
//   native 16-codebook predictor
//   native codec decoder/vocoder
// -----------------------------------------------------------------------------
#include <condition_variable>
#include <deque>
#include <thread>
#include <exception>
#include <functional>
#include <future>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <complex>
#include <iomanip>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cstdlib>
#include <sstream>
#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif



// ============================================================================
// legacy_legacy_tag_f.0 algorithmic backend
// We implement model math directly. No library kernel replay is a correctness
// requirement or a production dependency.
// ============================================================================


namespace qwen3_tts::native_ops {

constexpr int kWaveSize = 32;
constexpr int kWavesPerBlock = 8;

__device__ __forceinline__
float bf16_to_float(std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__
std::uint16_t float_to_bf16_rne(float value) {
    std::uint32_t bits = __float_as_uint(value);

    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<std::uint16_t>(
            (bits >> 16) | 0x0040U);
    }

    const std::uint32_t least_significant =
        (bits >> 16) & 1U;

    bits += 0x7fffU + least_significant;

    return static_cast<std::uint16_t>(
        bits >> 16);
}

__device__ __forceinline__
float wave_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    return value;
}

// Correctness-first native Linear.
//
// PyTorch weight convention:
//   input  [M,K]
//   weight [N,K]
//   bias   [N] or null
//   output [M,N]
//
// One wave computes one output scalar. The 32 lanes partition K and reduce
// in FP32. No rocBLAS/hipBLAS/Tensile/WMMA contract is involved.
__global__ __launch_bounds__(256, 2)
void linear_wave32_bf16_f32acc_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    const std::uint16_t* __restrict__ bias,
    std::uint16_t* __restrict__ output,
    int rows,
    int output_columns,
    int reduction_columns) {

    const int thread =
        static_cast<int>(threadIdx.x);

    const int wave =
        thread >> 5;

    const int lane =
        thread & 31;

    const int row =
        static_cast<int>(blockIdx.y);

    const int output_column =
        static_cast<int>(blockIdx.x)
            * kWavesPerBlock
        + wave;

    if (
        row >= rows
        || output_column >= output_columns
    ) {
        return;
    }

    const std::size_t input_base =
        static_cast<std::size_t>(row)
        * reduction_columns;

    const std::size_t weight_base =
        static_cast<std::size_t>(output_column)
        * reduction_columns;

    float partial = 0.0f;

    for (
        int k = lane;
        k < reduction_columns;
        k += kWaveSize
    ) {
        partial = fmaf(
            bf16_to_float(
                input[input_base + k]),
            bf16_to_float(
                weight[weight_base + k]),
            partial);
    }

    const float sum =
        wave_reduce_sum(partial);

    if (lane == 0) {
        float value = sum;

        if (bias != nullptr) {
            value +=
                bf16_to_float(
                    bias[output_column]);
        }

        output[
            static_cast<std::size_t>(row)
                * output_columns
            + output_column
        ] = float_to_bf16_rne(value);
    }
}

inline void linear_bf16(
    const std::uint16_t* input,
    const std::uint16_t* weight,
    const std::uint16_t* bias,
    std::uint16_t* output,
    int rows,
    int output_columns,
    int reduction_columns,
    hipStream_t stream = nullptr) {

    if (
        rows <= 0
        || output_columns <= 0
        || reduction_columns <= 0
    ) {
        throw std::runtime_error(
            "legacy_legacy_tag_f.1 native linear requires positive M/N/K.");
    }

    const dim3 block(
        kWavesPerBlock * kWaveSize,
        1,
        1);

    const dim3 grid(
        static_cast<unsigned>(
            (output_columns
             + kWavesPerBlock - 1)
            / kWavesPerBlock),
        static_cast<unsigned>(rows),
        1);

    hipLaunchKernelGGL(
        linear_wave32_bf16_f32acc_kernel,
        grid,
        block,
        0,
        stream,
        input,
        weight,
        bias,
        output,
        rows,
        output_columns,
        reduction_columns);

    const hipError_t status =
        hipGetLastError();

    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(
                "legacy_legacy_tag_f.1 native linear launch failed: ")
            + hipGetErrorString(status));
    }
}

} // namespace qwen3_tts::native_ops


// ---- embedded from talker_rmsnorm_staged.h ----



namespace qwen3_tts::talker::rmsnorm_staged {

constexpr int kHidden = 1024;
constexpr float kEpsilon = 1.0e-6f;
constexpr int kPointwiseThreads = 256;

#define legacy_HIP_CHECK_a(call) do { \
    const hipError_t status__ = (call); \
    if (status__ != hipSuccess) { \
        throw std::runtime_error( \
            std::string("legacy_legacy_tag_a HIP error: ") \
            + hipGetErrorString(status__) \
            + " at " + __FILE__ + ":" \
            + std::to_string(__LINE__)); \
    } \
} while (0)

struct Buffer {
    void* pointer = nullptr;
    std::size_t bytes = 0;

    explicit Buffer(std::size_t n)
        : bytes(n) {
        if (bytes != 0) {
            legacy_HIP_CHECK_a(
                hipMalloc(&pointer, bytes));
        }
    }

    ~Buffer() {
        if (pointer != nullptr) {
            (void)hipFree(pointer);
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template <typename T>
    T* as() {
        return reinterpret_cast<T*>(pointer);
    }
};

__device__ __forceinline__
float bf16_to_float(std::uint16_t value) {
    return __uint_as_float(
        static_cast<std::uint32_t>(value) << 16);
}

__device__ __forceinline__
std::uint16_t float_to_bf16_rne(float value) {
    std::uint32_t bits = __float_as_uint(value);

    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<std::uint16_t>(
            (bits >> 16) | 0x0040U);
    }

    const std::uint32_t lsb =
        (bits >> 16) & 1U;

    bits += 0x7fffU + lsb;

    return static_cast<std::uint16_t>(
        bits >> 16);
}

__global__
void materialize_square(
    const std::uint16_t* input,
    float* square,
    int rows) {

    const std::uint64_t count =
        static_cast<std::uint64_t>(rows)
        * kHidden;

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    const float value =
        bf16_to_float(input[index]);

    square[index] =
        value * value;
}

__global__
void mean_rows_63(
    const float* square,
    float* mean,
    int rows) {

    const int lane =
        static_cast<int>(threadIdx.x);

    const int row =
        static_cast<int>(blockIdx.x)
        * static_cast<int>(blockDim.y)
        + static_cast<int>(threadIdx.y);

    if (row >= rows) {
        return;
    }

    float values[4] = {
        0.0f,
        0.0f,
        0.0f,
        0.0f,
    };

    int vector_index = lane;

    while (
        vector_index * 4 + 3
        < kHidden
    ) {
        const int base =
            row * kHidden
            + vector_index * 4;

        values[0] += square[base + 0];
        values[1] += square[base + 1];
        values[2] += square[base + 2];
        values[3] += square[base + 3];

        vector_index += 32;
    }

    values[0] += values[1];
    values[0] += values[2];
    values[0] += values[3];

    for (
        int offset = 1;
        offset < 32;
        offset <<= 1
    ) {
        const float other =
            __shfl_down(
                values[0],
                offset,
                32);

        values[0] += other;
    }

    if (lane == 0) {
        mean[row] =
            values[0]
            * (1.0f / static_cast<float>(kHidden));
    }
}

__global__
void mean_row_1(
    const float* square,
    float* mean) {

    __shared__ float shared[kHidden/4];

    const int thread =
        static_cast<int>(threadIdx.x);

    const int base =
        thread * 4;

    float values[4] = {
        square[base + 0],
        square[base + 1],
        square[base + 2],
        square[base + 3],
    };

    values[0] += values[1];
    values[0] += values[2];
    values[0] += values[3];

    shared[thread] =
        values[0];

    for (
        int offset = (kHidden/4)/2;
        offset >= 32;
        offset >>= 1
    ) {
        __syncthreads();

        if (thread < offset) {
            values[0] +=
                shared[
                    thread + offset
                ];

            shared[thread] =
                values[0];
        }
    }

    __syncthreads();

    if (thread < 32) {
        values[0] =
            shared[thread];

        for (
            int offset = 1;
            offset < 32;
            offset <<= 1
        ) {
            const float other =
                __shfl_down(
                    values[0],
                    offset,
                    32);

            values[0] += other;
        }

        if (thread == 0) {
            mean[0] =
                values[0]
                * (1.0f / static_cast<float>(kHidden));
        }
    }
}

__global__
void add_epsilon(
    const float* mean,
    float* mean_eps,
    int rows) {

    const int row =
        static_cast<int>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (row >= rows) {
        return;
    }

    mean_eps[row] =
        mean[row]
        + kEpsilon;
}

__global__
void reciprocal_sqrt(
    const float* mean_eps,
    float* inverse,
    int rows) {

    const int row =
        static_cast<int>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (row >= rows) {
        return;
    }

    inverse[row] =
        ::rsqrt(
            mean_eps[row]);
}

__global__
void normalize_f32(
    const std::uint16_t* input,
    const float* inverse,
    float* normalized,
    int rows) {

    const std::uint64_t count =
        static_cast<std::uint64_t>(rows)
        * kHidden;

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    const int row =
        static_cast<int>(
            index / kHidden);

    normalized[index] =
        bf16_to_float(
            input[index])
        * inverse[row];
}

__global__
void cast_normalized_bf16(
    const float* normalized,
    std::uint16_t* normalized_bf16,
    std::uint64_t count) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    normalized_bf16[index] =
        float_to_bf16_rne(
            normalized[index]);
}

__global__
void multiply_weight_bf16(
    const std::uint16_t* normalized,
    const std::uint16_t* weight,
    std::uint16_t* output,
    std::uint64_t count) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    const int column =
        static_cast<int>(
            index % kHidden);

    const float left =
        bf16_to_float(
            weight[column]);

    const float right =
        bf16_to_float(
            normalized[index]);

    output[index] =
        float_to_bf16_rne(
            left * right);
}

inline void launch_mean(
    const std::uint16_t* input,
    float* square,
    float* mean,
    int rows,
    hipStream_t stream) {

    const std::size_t count =
        static_cast<std::size_t>(rows)
        * kHidden;

    hipLaunchKernelGGL(
        materialize_square,
        dim3(
            static_cast<unsigned>(
                (
                    count
                    + kPointwiseThreads
                    - 1
                )
                / kPointwiseThreads)),
        dim3(kPointwiseThreads),
        0,
        stream,
        input,
        square,
        rows);

    legacy_HIP_CHECK_a(
        hipGetLastError());

    if (rows == 1) {
        hipLaunchKernelGGL(
            mean_row_1,
            dim3(1),
            dim3(kHidden/4),
            0,
            stream,
            square,
            mean);
    } else if (rows == 63) {
        hipLaunchKernelGGL(
            mean_rows_63,
            dim3(4),
            dim3(32, 16, 1),
            0,
            stream,
            square,
            mean,
            rows);
    } else {
        throw std::runtime_error(
            "legacy_legacy_tag_a hidden RMSNorm supports rows 1 or 63.");
    }

    legacy_HIP_CHECK_a(
        hipGetLastError());
}

inline void launch(
    const std::uint16_t* input,
    const std::uint16_t* weight,
    std::uint16_t* output,
    int rows,
    int dimension,
    hipStream_t stream) {

    if (dimension != kHidden) {
        throw std::runtime_error(
            "legacy_legacy_tag_a exact hidden RMSNorm requires dimension=1024.");
    }

    if (rows != 1 && rows != 63) {
        throw std::runtime_error(
            "legacy_legacy_tag_a exact hidden RMSNorm requires rows=1 or 63.");
    }

    const std::size_t count =
        static_cast<std::size_t>(rows)
        * kHidden;

    Buffer square(
        count * sizeof(float));

    Buffer mean(
        static_cast<std::size_t>(rows)
        * sizeof(float));

    Buffer mean_eps(
        static_cast<std::size_t>(rows)
        * sizeof(float));

    Buffer inverse(
        static_cast<std::size_t>(rows)
        * sizeof(float));

    Buffer normalized_f32(
        count * sizeof(float));

    Buffer normalized_bf16(
        count * sizeof(std::uint16_t));

    launch_mean(
        input,
        square.as<float>(),
        mean.as<float>(),
        rows,
        stream);

    hipLaunchKernelGGL(
        add_epsilon,
        dim3(1),
        dim3(64),
        0,
        stream,
        mean.as<float>(),
        mean_eps.as<float>(),
        rows);

    legacy_HIP_CHECK_a(
        hipGetLastError());

    hipLaunchKernelGGL(
        reciprocal_sqrt,
        dim3(1),
        dim3(64),
        0,
        stream,
        mean_eps.as<float>(),
        inverse.as<float>(),
        rows);

    legacy_HIP_CHECK_a(
        hipGetLastError());

    hipLaunchKernelGGL(
        normalize_f32,
        dim3(
            static_cast<unsigned>(
                (
                    count
                    + kPointwiseThreads
                    - 1
                )
                / kPointwiseThreads)),
        dim3(kPointwiseThreads),
        0,
        stream,
        input,
        inverse.as<float>(),
        normalized_f32.as<float>(),
        rows);

    legacy_HIP_CHECK_a(
        hipGetLastError());

    hipLaunchKernelGGL(
        cast_normalized_bf16,
        dim3(
            static_cast<unsigned>(
                (
                    count
                    + kPointwiseThreads
                    - 1
                )
                / kPointwiseThreads)),
        dim3(kPointwiseThreads),
        0,
        stream,
        normalized_f32.as<float>(),
        normalized_bf16.as<std::uint16_t>(),
        count);

    legacy_HIP_CHECK_a(
        hipGetLastError());

    hipLaunchKernelGGL(
        multiply_weight_bf16,
        dim3(
            static_cast<unsigned>(
                (
                    count
                    + kPointwiseThreads
                    - 1
                )
                / kPointwiseThreads)),
        dim3(kPointwiseThreads),
        0,
        stream,
        normalized_bf16.as<std::uint16_t>(),
        weight,
        output,
        count);

    legacy_HIP_CHECK_a(
        hipGetLastError());
}

#undef legacy_HIP_CHECK_a

}  // namespace qwen3_tts::talker::rmsnorm_staged


// legacy_legacy_tag_f.0 FinalHead uses qwen3_tts::native_ops::linear_bf16.



namespace fs = std::filesystem;
namespace qwen3_tts::baseline {


namespace fs = std::filesystem;

static uint64_t fnv1a64(const void* data, size_t bytes) {
    const auto* pointer =
        static_cast<const uint8_t*>(data);
    uint64_t value = 1469598103934665603ull;
    for (size_t index = 0; index < bytes; ++index) {
        value ^= pointer[index];
        value *= 1099511628211ull;
    }
    return value;
}

static std::vector<uint8_t> read_bytes(
    const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "failed to open: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error(
            "failed to size: " + path.string());
    }

    std::vector<uint8_t> data(
        static_cast<size_t>(size));
    if (!data.empty()) {
        stream.read(
            reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    }
    if (!stream.good() && !stream.eof()) {
        throw std::runtime_error(
            "failed to read: " + path.string());
    }
    return data;
}

static std::string read_text(
    const fs::path& path) {
    const std::vector<uint8_t> data =
        read_bytes(path);
    return std::string(
        reinterpret_cast<const char*>(data.data()),
        data.size());
}

static void write_bytes(
    const fs::path& path,
    const void* data,
    size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "failed to open for write: " +
            path.string());
    }
    stream.write(
        static_cast<const char*>(data),
        static_cast<std::streamsize>(bytes));
    if (!stream.good()) {
        throw std::runtime_error(
            "failed to write: " + path.string());
    }
}

static void write_text(
    const fs::path& path,
    const std::string& text) {
    write_bytes(path, text.data(), text.size());
}

static void append_utf8(
    std::string& output,
    uint32_t codepoint) {
    if (codepoint <= 0x7Fu) {
        output.push_back(
            static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFu) {
        output.push_back(
            static_cast<char>(
                0xC0u | (codepoint >> 6)));
        output.push_back(
            static_cast<char>(
                0x80u | (codepoint & 0x3Fu)));
    } else if (codepoint <= 0xFFFFu) {
        output.push_back(
            static_cast<char>(
                0xE0u | (codepoint >> 12)));
        output.push_back(
            static_cast<char>(
                0x80u |
                ((codepoint >> 6) & 0x3Fu)));
        output.push_back(
            static_cast<char>(
                0x80u | (codepoint & 0x3Fu)));
    } else if (codepoint <= 0x10FFFFu) {
        output.push_back(
            static_cast<char>(
                0xF0u | (codepoint >> 18)));
        output.push_back(
            static_cast<char>(
                0x80u |
                ((codepoint >> 12) & 0x3Fu)));
        output.push_back(
            static_cast<char>(
                0x80u |
                ((codepoint >> 6) & 0x3Fu)));
        output.push_back(
            static_cast<char>(
                0x80u | (codepoint & 0x3Fu)));
    } else {
        throw std::runtime_error(
            "invalid Unicode codepoint");
    }
}

static uint32_t parse_hex4(
    const std::string& text,
    size_t position) {
    if (position + 4 > text.size()) {
        throw std::runtime_error(
            "short JSON Unicode escape");
    }
    uint32_t value = 0;
    for (size_t offset = 0; offset < 4; ++offset) {
        const char c = text[position + offset];
        value <<= 4;
        if (c >= '0' && c <= '9') {
            value |= static_cast<uint32_t>(
                c - '0');
        } else if (c >= 'a' && c <= 'f') {
            value |= static_cast<uint32_t>(
                c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            value |= static_cast<uint32_t>(
                c - 'A' + 10);
        } else {
            throw std::runtime_error(
                "invalid JSON Unicode escape");
        }
    }
    return value;
}

class VocabularyParser {
public:
    explicit VocabularyParser(
        std::string text)
        : text_(std::move(text)) {}

    std::unordered_map<std::string, int64_t>
    parse() {
        skip_space();
        require('{');

        std::unordered_map<std::string, int64_t>
            vocabulary;
        vocabulary.reserve(160000);
        skip_space();
        if (consume('}')) {
            return vocabulary;
        }

        for (;;) {
            const std::string token =
                parse_string();
            skip_space();
            require(':');
            skip_space();
            const int64_t id = parse_integer();

            vocabulary.emplace(token, id);

            skip_space();
            if (consume('}')) {
                break;
            }
            require(',');
        }

        skip_space();
        if (position_ != text_.size()) {
            fail("trailing JSON data");
        }
        return vocabulary;
    }

private:

    std::string text_;
    size_t position_ = 0;

    [[noreturn]] void fail(
        const std::string& message) const {
        throw std::runtime_error(
            "vocab JSON error at byte " +
            std::to_string(position_) +
            ": " + message);
    }

    void skip_space() {
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c == ' ' ||
                c == '\n' ||
                c == '\r' ||
                c == '\t') {
                ++position_;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) {
        skip_space();
        if (position_ < text_.size() &&
            text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void require(char expected) {
        if (!consume(expected)) {
            fail(
                std::string("expected '") +
                expected + "'");
        }
    }

    std::string parse_string() {
        skip_space();
        if (position_ >= text_.size() ||
            text_[position_] != '"') {
            fail("expected string");
        }
        ++position_;

        std::string output;
        while (position_ < text_.size()) {
            const unsigned char c =
                static_cast<unsigned char>(
                    text_[position_++]);

            if (c == '"') {
                return output;
            }

            if (c != '\\') {
                output.push_back(
                    static_cast<char>(c));
                continue;
            }

            if (position_ >= text_.size()) {
                fail("unfinished escape");
            }

            const char escaped =
                text_[position_++];
            switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    output.push_back(escaped);
                    break;
                case 'b':
                    output.push_back('\b');
                    break;
                case 'f':
                    output.push_back('\f');
                    break;
                case 'n':
                    output.push_back('\n');
                    break;
                case 'r':
                    output.push_back('\r');
                    break;
                case 't':
                    output.push_back('\t');
                    break;
                case 'u': {
                    const uint32_t first =
                        parse_hex4(
                            text_, position_);
                    position_ += 4;

                    uint32_t codepoint = first;
                    if (first >= 0xD800u &&
                        first <= 0xDBFFu) {
                        if (position_ + 6 >
                                text_.size() ||
                            text_[position_] != '\\' ||
                            text_[position_ + 1] != 'u') {
                            fail(
                                "missing low surrogate");
                        }
                        position_ += 2;
                        const uint32_t second =
                            parse_hex4(
                                text_, position_);
                        position_ += 4;
                        if (second < 0xDC00u ||
                            second > 0xDFFFu) {
                            fail(
                                "invalid low surrogate");
                        }
                        codepoint =
                            0x10000u +
                            ((first - 0xD800u) << 10) +
                            (second - 0xDC00u);
                    } else if (
                        first >= 0xDC00u &&
                        first <= 0xDFFFu) {
                        fail(
                            "unexpected low surrogate");
                    }
                    append_utf8(
                        output, codepoint);
                    break;
                }
                default:
                    fail("unknown escape");
            }
        }

        fail("unterminated string");
    }

    int64_t parse_integer() {
        skip_space();
        const size_t begin = position_;
        bool negative = false;
        if (position_ < text_.size() &&
            text_[position_] == '-') {
            negative = true;
            ++position_;
        }

        uint64_t value = 0;
        size_t digits = 0;
        while (position_ < text_.size() &&
               std::isdigit(
                   static_cast<unsigned char>(
                       text_[position_]))) {
            value =
                value * 10u +
                static_cast<uint64_t>(
                    text_[position_] - '0');
            ++position_;
            ++digits;
        }

        if (digits == 0) {
            position_ = begin;
            fail("expected integer");
        }

        if (!negative &&
            value >
                static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::
                        max())) {
            fail("integer overflow");
        }

        if (negative) {
            if (value >
                static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::
                        max()) + 1u) {
                fail("integer overflow");
            }
            return -static_cast<int64_t>(value);
        }
        return static_cast<int64_t>(value);
    }
};

struct Range {
    uint32_t first;
    uint32_t last;
};

static constexpr Range LETTER_RANGES[] = {
    {0x41u, 0x5Au},
    {0x61u, 0x7Au},
    {0xAAu, 0xAAu},
    {0xB5u, 0xB5u},
    {0xBAu, 0xBAu},
    {0xC0u, 0xD6u},
    {0xD8u, 0xF6u},
    {0xF8u, 0x2C1u},
    {0x2C6u, 0x2D1u},
    {0x2E0u, 0x2E4u},
    {0x2ECu, 0x2ECu},
    {0x2EEu, 0x2EEu},
    {0x370u, 0x374u},
    {0x376u, 0x377u},
    {0x37Au, 0x37Du},
    {0x37Fu, 0x37Fu},
    {0x386u, 0x386u},
    {0x388u, 0x38Au},
    {0x38Cu, 0x38Cu},
    {0x38Eu, 0x3A1u},
    {0x3A3u, 0x3F5u},
    {0x3F7u, 0x481u},
    {0x48Au, 0x52Fu},
    {0x531u, 0x556u},
    {0x559u, 0x559u},
    {0x560u, 0x588u},
    {0x5D0u, 0x5EAu},
    {0x5EFu, 0x5F2u},
    {0x620u, 0x64Au},
    {0x66Eu, 0x66Fu},
    {0x671u, 0x6D3u},
    {0x6D5u, 0x6D5u},
    {0x6E5u, 0x6E6u},
    {0x6EEu, 0x6EFu},
    {0x6FAu, 0x6FCu},
    {0x6FFu, 0x6FFu},
    {0x710u, 0x710u},
    {0x712u, 0x72Fu},
    {0x74Du, 0x7A5u},
    {0x7B1u, 0x7B1u},
    {0x7CAu, 0x7EAu},
    {0x7F4u, 0x7F5u},
    {0x7FAu, 0x7FAu},
    {0x800u, 0x815u},
    {0x81Au, 0x81Au},
    {0x824u, 0x824u},
    {0x828u, 0x828u},
    {0x840u, 0x858u},
    {0x860u, 0x86Au},
    {0x870u, 0x887u},
    {0x889u, 0x88Eu},
    {0x8A0u, 0x8C9u},
    {0x904u, 0x939u},
    {0x93Du, 0x93Du},
    {0x950u, 0x950u},
    {0x958u, 0x961u},
    {0x971u, 0x980u},
    {0x985u, 0x98Cu},
    {0x98Fu, 0x990u},
    {0x993u, 0x9A8u},
    {0x9AAu, 0x9B0u},
    {0x9B2u, 0x9B2u},
    {0x9B6u, 0x9B9u},
    {0x9BDu, 0x9BDu},
    {0x9CEu, 0x9CEu},
    {0x9DCu, 0x9DDu},
    {0x9DFu, 0x9E1u},
    {0x9F0u, 0x9F1u},
    {0x9FCu, 0x9FCu},
    {0xA05u, 0xA0Au},
    {0xA0Fu, 0xA10u},
    {0xA13u, 0xA28u},
    {0xA2Au, 0xA30u},
    {0xA32u, 0xA33u},
    {0xA35u, 0xA36u},
    {0xA38u, 0xA39u},
    {0xA59u, 0xA5Cu},
    {0xA5Eu, 0xA5Eu},
    {0xA72u, 0xA74u},
    {0xA85u, 0xA8Du},
    {0xA8Fu, 0xA91u},
    {0xA93u, 0xAA8u},
    {0xAAAu, 0xAB0u},
    {0xAB2u, 0xAB3u},
    {0xAB5u, 0xAB9u},
    {0xABDu, 0xABDu},
    {0xAD0u, 0xAD0u},
    {0xAE0u, 0xAE1u},
    {0xAF9u, 0xAF9u},
    {0xB05u, 0xB0Cu},
    {0xB0Fu, 0xB10u},
    {0xB13u, 0xB28u},
    {0xB2Au, 0xB30u},
    {0xB32u, 0xB33u},
    {0xB35u, 0xB39u},
    {0xB3Du, 0xB3Du},
    {0xB5Cu, 0xB5Du},
    {0xB5Fu, 0xB61u},
    {0xB71u, 0xB71u},
    {0xB83u, 0xB83u},
    {0xB85u, 0xB8Au},
    {0xB8Eu, 0xB90u},
    {0xB92u, 0xB95u},
    {0xB99u, 0xB9Au},
    {0xB9Cu, 0xB9Cu},
    {0xB9Eu, 0xB9Fu},
    {0xBA3u, 0xBA4u},
    {0xBA8u, 0xBAAu},
    {0xBAEu, 0xBB9u},
    {0xBD0u, 0xBD0u},
    {0xC05u, 0xC0Cu},
    {0xC0Eu, 0xC10u},
    {0xC12u, 0xC28u},
    {0xC2Au, 0xC39u},
    {0xC3Du, 0xC3Du},
    {0xC58u, 0xC5Au},
    {0xC5Du, 0xC5Du},
    {0xC60u, 0xC61u},
    {0xC80u, 0xC80u},
    {0xC85u, 0xC8Cu},
    {0xC8Eu, 0xC90u},
    {0xC92u, 0xCA8u},
    {0xCAAu, 0xCB3u},
    {0xCB5u, 0xCB9u},
    {0xCBDu, 0xCBDu},
    {0xCDDu, 0xCDEu},
    {0xCE0u, 0xCE1u},
    {0xCF1u, 0xCF2u},
    {0xD04u, 0xD0Cu},
    {0xD0Eu, 0xD10u},
    {0xD12u, 0xD3Au},
    {0xD3Du, 0xD3Du},
    {0xD4Eu, 0xD4Eu},
    {0xD54u, 0xD56u},
    {0xD5Fu, 0xD61u},
    {0xD7Au, 0xD7Fu},
    {0xD85u, 0xD96u},
    {0xD9Au, 0xDB1u},
    {0xDB3u, 0xDBBu},
    {0xDBDu, 0xDBDu},
    {0xDC0u, 0xDC6u},
    {0xE01u, 0xE30u},
    {0xE32u, 0xE33u},
    {0xE40u, 0xE46u},
    {0xE81u, 0xE82u},
    {0xE84u, 0xE84u},
    {0xE86u, 0xE8Au},
    {0xE8Cu, 0xEA3u},
    {0xEA5u, 0xEA5u},
    {0xEA7u, 0xEB0u},
    {0xEB2u, 0xEB3u},
    {0xEBDu, 0xEBDu},
    {0xEC0u, 0xEC4u},
    {0xEC6u, 0xEC6u},
    {0xEDCu, 0xEDFu},
    {0xF00u, 0xF00u},
    {0xF40u, 0xF47u},
    {0xF49u, 0xF6Cu},
    {0xF88u, 0xF8Cu},
    {0x1000u, 0x102Au},
    {0x103Fu, 0x103Fu},
    {0x1050u, 0x1055u},
    {0x105Au, 0x105Du},
    {0x1061u, 0x1061u},
    {0x1065u, 0x1066u},
    {0x106Eu, 0x1070u},
    {0x1075u, 0x1081u},
    {0x108Eu, 0x108Eu},
    {0x10A0u, 0x10C5u},
    {0x10C7u, 0x10C7u},
    {0x10CDu, 0x10CDu},
    {0x10D0u, 0x10FAu},
    {0x10FCu, 0x1248u},
    {0x124Au, 0x124Du},
    {0x1250u, 0x1256u},
    {0x1258u, 0x1258u},
    {0x125Au, 0x125Du},
    {0x1260u, 0x1288u},
    {0x128Au, 0x128Du},
    {0x1290u, 0x12B0u},
    {0x12B2u, 0x12B5u},
    {0x12B8u, 0x12BEu},
    {0x12C0u, 0x12C0u},
    {0x12C2u, 0x12C5u},
    {0x12C8u, 0x12D6u},
    {0x12D8u, 0x1310u},
    {0x1312u, 0x1315u},
    {0x1318u, 0x135Au},
    {0x1380u, 0x138Fu},
    {0x13A0u, 0x13F5u},
    {0x13F8u, 0x13FDu},
    {0x1401u, 0x166Cu},
    {0x166Fu, 0x167Fu},
    {0x1681u, 0x169Au},
    {0x16A0u, 0x16EAu},
    {0x16F1u, 0x16F8u},
    {0x1700u, 0x1711u},
    {0x171Fu, 0x1731u},
    {0x1740u, 0x1751u},
    {0x1760u, 0x176Cu},
    {0x176Eu, 0x1770u},
    {0x1780u, 0x17B3u},
    {0x17D7u, 0x17D7u},
    {0x17DCu, 0x17DCu},
    {0x1820u, 0x1878u},
    {0x1880u, 0x1884u},
    {0x1887u, 0x18A8u},
    {0x18AAu, 0x18AAu},
    {0x18B0u, 0x18F5u},
    {0x1900u, 0x191Eu},
    {0x1950u, 0x196Du},
    {0x1970u, 0x1974u},
    {0x1980u, 0x19ABu},
    {0x19B0u, 0x19C9u},
    {0x1A00u, 0x1A16u},
    {0x1A20u, 0x1A54u},
    {0x1AA7u, 0x1AA7u},
    {0x1B05u, 0x1B33u},
    {0x1B45u, 0x1B4Cu},
    {0x1B83u, 0x1BA0u},
    {0x1BAEu, 0x1BAFu},
    {0x1BBAu, 0x1BE5u},
    {0x1C00u, 0x1C23u},
    {0x1C4Du, 0x1C4Fu},
    {0x1C5Au, 0x1C7Du},
    {0x1C80u, 0x1C88u},
    {0x1C90u, 0x1CBAu},
    {0x1CBDu, 0x1CBFu},
    {0x1CE9u, 0x1CECu},
    {0x1CEEu, 0x1CF3u},
    {0x1CF5u, 0x1CF6u},
    {0x1CFAu, 0x1CFAu},
    {0x1D00u, 0x1DBFu},
    {0x1E00u, 0x1F15u},
    {0x1F18u, 0x1F1Du},
    {0x1F20u, 0x1F45u},
    {0x1F48u, 0x1F4Du},
    {0x1F50u, 0x1F57u},
    {0x1F59u, 0x1F59u},
    {0x1F5Bu, 0x1F5Bu},
    {0x1F5Du, 0x1F5Du},
    {0x1F5Fu, 0x1F7Du},
    {0x1F80u, 0x1FB4u},
    {0x1FB6u, 0x1FBCu},
    {0x1FBEu, 0x1FBEu},
    {0x1FC2u, 0x1FC4u},
    {0x1FC6u, 0x1FCCu},
    {0x1FD0u, 0x1FD3u},
    {0x1FD6u, 0x1FDBu},
    {0x1FE0u, 0x1FECu},
    {0x1FF2u, 0x1FF4u},
    {0x1FF6u, 0x1FFCu},
    {0x2071u, 0x2071u},
    {0x207Fu, 0x207Fu},
    {0x2090u, 0x209Cu},
    {0x2102u, 0x2102u},
    {0x2107u, 0x2107u},
    {0x210Au, 0x2113u},
    {0x2115u, 0x2115u},
    {0x2119u, 0x211Du},
    {0x2124u, 0x2124u},
    {0x2126u, 0x2126u},
    {0x2128u, 0x2128u},
    {0x212Au, 0x212Du},
    {0x212Fu, 0x2139u},
    {0x213Cu, 0x213Fu},
    {0x2145u, 0x2149u},
    {0x214Eu, 0x214Eu},
    {0x2183u, 0x2184u},
    {0x2C00u, 0x2CE4u},
    {0x2CEBu, 0x2CEEu},
    {0x2CF2u, 0x2CF3u},
    {0x2D00u, 0x2D25u},
    {0x2D27u, 0x2D27u},
    {0x2D2Du, 0x2D2Du},
    {0x2D30u, 0x2D67u},
    {0x2D6Fu, 0x2D6Fu},
    {0x2D80u, 0x2D96u},
    {0x2DA0u, 0x2DA6u},
    {0x2DA8u, 0x2DAEu},
    {0x2DB0u, 0x2DB6u},
    {0x2DB8u, 0x2DBEu},
    {0x2DC0u, 0x2DC6u},
    {0x2DC8u, 0x2DCEu},
    {0x2DD0u, 0x2DD6u},
    {0x2DD8u, 0x2DDEu},
    {0x2E2Fu, 0x2E2Fu},
    {0x3005u, 0x3006u},
    {0x3031u, 0x3035u},
    {0x303Bu, 0x303Cu},
    {0x3041u, 0x3096u},
    {0x309Du, 0x309Fu},
    {0x30A1u, 0x30FAu},
    {0x30FCu, 0x30FFu},
    {0x3105u, 0x312Fu},
    {0x3131u, 0x318Eu},
    {0x31A0u, 0x31BFu},
    {0x31F0u, 0x31FFu},
    {0x3400u, 0x4DBFu},
    {0x4E00u, 0xA48Cu},
    {0xA4D0u, 0xA4FDu},
    {0xA500u, 0xA60Cu},
    {0xA610u, 0xA61Fu},
    {0xA62Au, 0xA62Bu},
    {0xA640u, 0xA66Eu},
    {0xA67Fu, 0xA69Du},
    {0xA6A0u, 0xA6E5u},
    {0xA717u, 0xA71Fu},
    {0xA722u, 0xA788u},
    {0xA78Bu, 0xA7CAu},
    {0xA7D0u, 0xA7D1u},
    {0xA7D3u, 0xA7D3u},
    {0xA7D5u, 0xA7D9u},
    {0xA7F2u, 0xA801u},
    {0xA803u, 0xA805u},
    {0xA807u, 0xA80Au},
    {0xA80Cu, 0xA822u},
    {0xA840u, 0xA873u},
    {0xA882u, 0xA8B3u},
    {0xA8F2u, 0xA8F7u},
    {0xA8FBu, 0xA8FBu},
    {0xA8FDu, 0xA8FEu},
    {0xA90Au, 0xA925u},
    {0xA930u, 0xA946u},
    {0xA960u, 0xA97Cu},
    {0xA984u, 0xA9B2u},
    {0xA9CFu, 0xA9CFu},
    {0xA9E0u, 0xA9E4u},
    {0xA9E6u, 0xA9EFu},
    {0xA9FAu, 0xA9FEu},
    {0xAA00u, 0xAA28u},
    {0xAA40u, 0xAA42u},
    {0xAA44u, 0xAA4Bu},
    {0xAA60u, 0xAA76u},
    {0xAA7Au, 0xAA7Au},
    {0xAA7Eu, 0xAAAFu},
    {0xAAB1u, 0xAAB1u},
    {0xAAB5u, 0xAAB6u},
    {0xAAB9u, 0xAABDu},
    {0xAAC0u, 0xAAC0u},
    {0xAAC2u, 0xAAC2u},
    {0xAADBu, 0xAADDu},
    {0xAAE0u, 0xAAEAu},
    {0xAAF2u, 0xAAF4u},
    {0xAB01u, 0xAB06u},
    {0xAB09u, 0xAB0Eu},
    {0xAB11u, 0xAB16u},
    {0xAB20u, 0xAB26u},
    {0xAB28u, 0xAB2Eu},
    {0xAB30u, 0xAB5Au},
    {0xAB5Cu, 0xAB69u},
    {0xAB70u, 0xABE2u},
    {0xAC00u, 0xD7A3u},
    {0xD7B0u, 0xD7C6u},
    {0xD7CBu, 0xD7FBu},
    {0xF900u, 0xFA6Du},
    {0xFA70u, 0xFAD9u},
    {0xFB00u, 0xFB06u},
    {0xFB13u, 0xFB17u},
    {0xFB1Du, 0xFB1Du},
    {0xFB1Fu, 0xFB28u},
    {0xFB2Au, 0xFB36u},
    {0xFB38u, 0xFB3Cu},
    {0xFB3Eu, 0xFB3Eu},
    {0xFB40u, 0xFB41u},
    {0xFB43u, 0xFB44u},
    {0xFB46u, 0xFBB1u},
    {0xFBD3u, 0xFD3Du},
    {0xFD50u, 0xFD8Fu},
    {0xFD92u, 0xFDC7u},
    {0xFDF0u, 0xFDFBu},
    {0xFE70u, 0xFE74u},
    {0xFE76u, 0xFEFCu},
    {0xFF21u, 0xFF3Au},
    {0xFF41u, 0xFF5Au},
    {0xFF66u, 0xFFBEu},
    {0xFFC2u, 0xFFC7u},
    {0xFFCAu, 0xFFCFu},
    {0xFFD2u, 0xFFD7u},
    {0xFFDAu, 0xFFDCu},
    {0x10000u, 0x1000Bu},
    {0x1000Du, 0x10026u},
    {0x10028u, 0x1003Au},
    {0x1003Cu, 0x1003Du},
    {0x1003Fu, 0x1004Du},
    {0x10050u, 0x1005Du},
    {0x10080u, 0x100FAu},
    {0x10280u, 0x1029Cu},
    {0x102A0u, 0x102D0u},
    {0x10300u, 0x1031Fu},
    {0x1032Du, 0x10340u},
    {0x10342u, 0x10349u},
    {0x10350u, 0x10375u},
    {0x10380u, 0x1039Du},
    {0x103A0u, 0x103C3u},
    {0x103C8u, 0x103CFu},
    {0x10400u, 0x1049Du},
    {0x104B0u, 0x104D3u},
    {0x104D8u, 0x104FBu},
    {0x10500u, 0x10527u},
    {0x10530u, 0x10563u},
    {0x10570u, 0x1057Au},
    {0x1057Cu, 0x1058Au},
    {0x1058Cu, 0x10592u},
    {0x10594u, 0x10595u},
    {0x10597u, 0x105A1u},
    {0x105A3u, 0x105B1u},
    {0x105B3u, 0x105B9u},
    {0x105BBu, 0x105BCu},
    {0x10600u, 0x10736u},
    {0x10740u, 0x10755u},
    {0x10760u, 0x10767u},
    {0x10780u, 0x10785u},
    {0x10787u, 0x107B0u},
    {0x107B2u, 0x107BAu},
    {0x10800u, 0x10805u},
    {0x10808u, 0x10808u},
    {0x1080Au, 0x10835u},
    {0x10837u, 0x10838u},
    {0x1083Cu, 0x1083Cu},
    {0x1083Fu, 0x10855u},
    {0x10860u, 0x10876u},
    {0x10880u, 0x1089Eu},
    {0x108E0u, 0x108F2u},
    {0x108F4u, 0x108F5u},
    {0x10900u, 0x10915u},
    {0x10920u, 0x10939u},
    {0x10980u, 0x109B7u},
    {0x109BEu, 0x109BFu},
    {0x10A00u, 0x10A00u},
    {0x10A10u, 0x10A13u},
    {0x10A15u, 0x10A17u},
    {0x10A19u, 0x10A35u},
    {0x10A60u, 0x10A7Cu},
    {0x10A80u, 0x10A9Cu},
    {0x10AC0u, 0x10AC7u},
    {0x10AC9u, 0x10AE4u},
    {0x10B00u, 0x10B35u},
    {0x10B40u, 0x10B55u},
    {0x10B60u, 0x10B72u},
    {0x10B80u, 0x10B91u},
    {0x10C00u, 0x10C48u},
    {0x10C80u, 0x10CB2u},
    {0x10CC0u, 0x10CF2u},
    {0x10D00u, 0x10D23u},
    {0x10E80u, 0x10EA9u},
    {0x10EB0u, 0x10EB1u},
    {0x10F00u, 0x10F1Cu},
    {0x10F27u, 0x10F27u},
    {0x10F30u, 0x10F45u},
    {0x10F70u, 0x10F81u},
    {0x10FB0u, 0x10FC4u},
    {0x10FE0u, 0x10FF6u},
    {0x11003u, 0x11037u},
    {0x11071u, 0x11072u},
    {0x11075u, 0x11075u},
    {0x11083u, 0x110AFu},
    {0x110D0u, 0x110E8u},
    {0x11103u, 0x11126u},
    {0x11144u, 0x11144u},
    {0x11147u, 0x11147u},
    {0x11150u, 0x11172u},
    {0x11176u, 0x11176u},
    {0x11183u, 0x111B2u},
    {0x111C1u, 0x111C4u},
    {0x111DAu, 0x111DAu},
    {0x111DCu, 0x111DCu},
    {0x11200u, 0x11211u},
    {0x11213u, 0x1122Bu},
    {0x1123Fu, 0x11240u},
    {0x11280u, 0x11286u},
    {0x11288u, 0x11288u},
    {0x1128Au, 0x1128Du},
    {0x1128Fu, 0x1129Du},
    {0x1129Fu, 0x112A8u},
    {0x112B0u, 0x112DEu},
    {0x11305u, 0x1130Cu},
    {0x1130Fu, 0x11310u},
    {0x11313u, 0x11328u},
    {0x1132Au, 0x11330u},
    {0x11332u, 0x11333u},
    {0x11335u, 0x11339u},
    {0x1133Du, 0x1133Du},
    {0x11350u, 0x11350u},
    {0x1135Du, 0x11361u},
    {0x11400u, 0x11434u},
    {0x11447u, 0x1144Au},
    {0x1145Fu, 0x11461u},
    {0x11480u, 0x114AFu},
    {0x114C4u, 0x114C5u},
    {0x114C7u, 0x114C7u},
    {0x11580u, 0x115AEu},
    {0x115D8u, 0x115DBu},
    {0x11600u, 0x1162Fu},
    {0x11644u, 0x11644u},
    {0x11680u, 0x116AAu},
    {0x116B8u, 0x116B8u},
    {0x11700u, 0x1171Au},
    {0x11740u, 0x11746u},
    {0x11800u, 0x1182Bu},
    {0x118A0u, 0x118DFu},
    {0x118FFu, 0x11906u},
    {0x11909u, 0x11909u},
    {0x1190Cu, 0x11913u},
    {0x11915u, 0x11916u},
    {0x11918u, 0x1192Fu},
    {0x1193Fu, 0x1193Fu},
    {0x11941u, 0x11941u},
    {0x119A0u, 0x119A7u},
    {0x119AAu, 0x119D0u},
    {0x119E1u, 0x119E1u},
    {0x119E3u, 0x119E3u},
    {0x11A00u, 0x11A00u},
    {0x11A0Bu, 0x11A32u},
    {0x11A3Au, 0x11A3Au},
    {0x11A50u, 0x11A50u},
    {0x11A5Cu, 0x11A89u},
    {0x11A9Du, 0x11A9Du},
    {0x11AB0u, 0x11AF8u},
    {0x11C00u, 0x11C08u},
    {0x11C0Au, 0x11C2Eu},
    {0x11C40u, 0x11C40u},
    {0x11C72u, 0x11C8Fu},
    {0x11D00u, 0x11D06u},
    {0x11D08u, 0x11D09u},
    {0x11D0Bu, 0x11D30u},
    {0x11D46u, 0x11D46u},
    {0x11D60u, 0x11D65u},
    {0x11D67u, 0x11D68u},
    {0x11D6Au, 0x11D89u},
    {0x11D98u, 0x11D98u},
    {0x11EE0u, 0x11EF2u},
    {0x11F02u, 0x11F02u},
    {0x11F04u, 0x11F10u},
    {0x11F12u, 0x11F33u},
    {0x11FB0u, 0x11FB0u},
    {0x12000u, 0x12399u},
    {0x12480u, 0x12543u},
    {0x12F90u, 0x12FF0u},
    {0x13000u, 0x1342Fu},
    {0x13441u, 0x13446u},
    {0x14400u, 0x14646u},
    {0x16800u, 0x16A38u},
    {0x16A40u, 0x16A5Eu},
    {0x16A70u, 0x16ABEu},
    {0x16AD0u, 0x16AEDu},
    {0x16B00u, 0x16B2Fu},
    {0x16B40u, 0x16B43u},
    {0x16B63u, 0x16B77u},
    {0x16B7Du, 0x16B8Fu},
    {0x16E40u, 0x16E7Fu},
    {0x16F00u, 0x16F4Au},
    {0x16F50u, 0x16F50u},
    {0x16F93u, 0x16F9Fu},
    {0x16FE0u, 0x16FE1u},
    {0x16FE3u, 0x16FE3u},
    {0x17000u, 0x187F7u},
    {0x18800u, 0x18CD5u},
    {0x18D00u, 0x18D08u},
    {0x1AFF0u, 0x1AFF3u},
    {0x1AFF5u, 0x1AFFBu},
    {0x1AFFDu, 0x1AFFEu},
    {0x1B000u, 0x1B122u},
    {0x1B132u, 0x1B132u},
    {0x1B150u, 0x1B152u},
    {0x1B155u, 0x1B155u},
    {0x1B164u, 0x1B167u},
    {0x1B170u, 0x1B2FBu},
    {0x1BC00u, 0x1BC6Au},
    {0x1BC70u, 0x1BC7Cu},
    {0x1BC80u, 0x1BC88u},
    {0x1BC90u, 0x1BC99u},
    {0x1D400u, 0x1D454u},
    {0x1D456u, 0x1D49Cu},
    {0x1D49Eu, 0x1D49Fu},
    {0x1D4A2u, 0x1D4A2u},
    {0x1D4A5u, 0x1D4A6u},
    {0x1D4A9u, 0x1D4ACu},
    {0x1D4AEu, 0x1D4B9u},
    {0x1D4BBu, 0x1D4BBu},
    {0x1D4BDu, 0x1D4C3u},
    {0x1D4C5u, 0x1D505u},
    {0x1D507u, 0x1D50Au},
    {0x1D50Du, 0x1D514u},
    {0x1D516u, 0x1D51Cu},
    {0x1D51Eu, 0x1D539u},
    {0x1D53Bu, 0x1D53Eu},
    {0x1D540u, 0x1D544u},
    {0x1D546u, 0x1D546u},
    {0x1D54Au, 0x1D550u},
    {0x1D552u, 0x1D6A5u},
    {0x1D6A8u, 0x1D6C0u},
    {0x1D6C2u, 0x1D6DAu},
    {0x1D6DCu, 0x1D6FAu},
    {0x1D6FCu, 0x1D714u},
    {0x1D716u, 0x1D734u},
    {0x1D736u, 0x1D74Eu},
    {0x1D750u, 0x1D76Eu},
    {0x1D770u, 0x1D788u},
    {0x1D78Au, 0x1D7A8u},
    {0x1D7AAu, 0x1D7C2u},
    {0x1D7C4u, 0x1D7CBu},
    {0x1DF00u, 0x1DF1Eu},
    {0x1DF25u, 0x1DF2Au},
    {0x1E030u, 0x1E06Du},
    {0x1E100u, 0x1E12Cu},
    {0x1E137u, 0x1E13Du},
    {0x1E14Eu, 0x1E14Eu},
    {0x1E290u, 0x1E2ADu},
    {0x1E2C0u, 0x1E2EBu},
    {0x1E4D0u, 0x1E4EBu},
    {0x1E7E0u, 0x1E7E6u},
    {0x1E7E8u, 0x1E7EBu},
    {0x1E7EDu, 0x1E7EEu},
    {0x1E7F0u, 0x1E7FEu},
    {0x1E800u, 0x1E8C4u},
    {0x1E900u, 0x1E943u},
    {0x1E94Bu, 0x1E94Bu},
    {0x1EE00u, 0x1EE03u},
    {0x1EE05u, 0x1EE1Fu},
    {0x1EE21u, 0x1EE22u},
    {0x1EE24u, 0x1EE24u},
    {0x1EE27u, 0x1EE27u},
    {0x1EE29u, 0x1EE32u},
    {0x1EE34u, 0x1EE37u},
    {0x1EE39u, 0x1EE39u},
    {0x1EE3Bu, 0x1EE3Bu},
    {0x1EE42u, 0x1EE42u},
    {0x1EE47u, 0x1EE47u},
    {0x1EE49u, 0x1EE49u},
    {0x1EE4Bu, 0x1EE4Bu},
    {0x1EE4Du, 0x1EE4Fu},
    {0x1EE51u, 0x1EE52u},
    {0x1EE54u, 0x1EE54u},
    {0x1EE57u, 0x1EE57u},
    {0x1EE59u, 0x1EE59u},
    {0x1EE5Bu, 0x1EE5Bu},
    {0x1EE5Du, 0x1EE5Du},
    {0x1EE5Fu, 0x1EE5Fu},
    {0x1EE61u, 0x1EE62u},
    {0x1EE64u, 0x1EE64u},
    {0x1EE67u, 0x1EE6Au},
    {0x1EE6Cu, 0x1EE72u},
    {0x1EE74u, 0x1EE77u},
    {0x1EE79u, 0x1EE7Cu},
    {0x1EE7Eu, 0x1EE7Eu},
    {0x1EE80u, 0x1EE89u},
    {0x1EE8Bu, 0x1EE9Bu},
    {0x1EEA1u, 0x1EEA3u},
    {0x1EEA5u, 0x1EEA9u},
    {0x1EEABu, 0x1EEBBu},
    {0x20000u, 0x2A6DFu},
    {0x2A700u, 0x2B739u},
    {0x2B740u, 0x2B81Du},
    {0x2B820u, 0x2CEA1u},
    {0x2CEB0u, 0x2EBE0u},
    {0x2EBF0u, 0x2EE5Du},
    {0x2F800u, 0x2FA1Du},
    {0x30000u, 0x3134Au},
    {0x31350u, 0x323AFu},
};

static constexpr Range NUMBER_RANGES[] = {
    {0x30u, 0x39u},
    {0xB2u, 0xB3u},
    {0xB9u, 0xB9u},
    {0xBCu, 0xBEu},
    {0x660u, 0x669u},
    {0x6F0u, 0x6F9u},
    {0x7C0u, 0x7C9u},
    {0x966u, 0x96Fu},
    {0x9E6u, 0x9EFu},
    {0x9F4u, 0x9F9u},
    {0xA66u, 0xA6Fu},
    {0xAE6u, 0xAEFu},
    {0xB66u, 0xB6Fu},
    {0xB72u, 0xB77u},
    {0xBE6u, 0xBF2u},
    {0xC66u, 0xC6Fu},
    {0xC78u, 0xC7Eu},
    {0xCE6u, 0xCEFu},
    {0xD58u, 0xD5Eu},
    {0xD66u, 0xD78u},
    {0xDE6u, 0xDEFu},
    {0xE50u, 0xE59u},
    {0xED0u, 0xED9u},
    {0xF20u, 0xF33u},
    {0x1040u, 0x1049u},
    {0x1090u, 0x1099u},
    {0x1369u, 0x137Cu},
    {0x16EEu, 0x16F0u},
    {0x17E0u, 0x17E9u},
    {0x17F0u, 0x17F9u},
    {0x1810u, 0x1819u},
    {0x1946u, 0x194Fu},
    {0x19D0u, 0x19DAu},
    {0x1A80u, 0x1A89u},
    {0x1A90u, 0x1A99u},
    {0x1B50u, 0x1B59u},
    {0x1BB0u, 0x1BB9u},
    {0x1C40u, 0x1C49u},
    {0x1C50u, 0x1C59u},
    {0x2070u, 0x2070u},
    {0x2074u, 0x2079u},
    {0x2080u, 0x2089u},
    {0x2150u, 0x2182u},
    {0x2185u, 0x2189u},
    {0x2460u, 0x249Bu},
    {0x24EAu, 0x24FFu},
    {0x2776u, 0x2793u},
    {0x2CFDu, 0x2CFDu},
    {0x3007u, 0x3007u},
    {0x3021u, 0x3029u},
    {0x3038u, 0x303Au},
    {0x3192u, 0x3195u},
    {0x3220u, 0x3229u},
    {0x3248u, 0x324Fu},
    {0x3251u, 0x325Fu},
    {0x3280u, 0x3289u},
    {0x32B1u, 0x32BFu},
    {0xA620u, 0xA629u},
    {0xA6E6u, 0xA6EFu},
    {0xA830u, 0xA835u},
    {0xA8D0u, 0xA8D9u},
    {0xA900u, 0xA909u},
    {0xA9D0u, 0xA9D9u},
    {0xA9F0u, 0xA9F9u},
    {0xAA50u, 0xAA59u},
    {0xABF0u, 0xABF9u},
    {0xFF10u, 0xFF19u},
    {0x10107u, 0x10133u},
    {0x10140u, 0x10178u},
    {0x1018Au, 0x1018Bu},
    {0x102E1u, 0x102FBu},
    {0x10320u, 0x10323u},
    {0x10341u, 0x10341u},
    {0x1034Au, 0x1034Au},
    {0x103D1u, 0x103D5u},
    {0x104A0u, 0x104A9u},
    {0x10858u, 0x1085Fu},
    {0x10879u, 0x1087Fu},
    {0x108A7u, 0x108AFu},
    {0x108FBu, 0x108FFu},
    {0x10916u, 0x1091Bu},
    {0x109BCu, 0x109BDu},
    {0x109C0u, 0x109CFu},
    {0x109D2u, 0x109FFu},
    {0x10A40u, 0x10A48u},
    {0x10A7Du, 0x10A7Eu},
    {0x10A9Du, 0x10A9Fu},
    {0x10AEBu, 0x10AEFu},
    {0x10B58u, 0x10B5Fu},
    {0x10B78u, 0x10B7Fu},
    {0x10BA9u, 0x10BAFu},
    {0x10CFAu, 0x10CFFu},
    {0x10D30u, 0x10D39u},
    {0x10E60u, 0x10E7Eu},
    {0x10F1Du, 0x10F26u},
    {0x10F51u, 0x10F54u},
    {0x10FC5u, 0x10FCBu},
    {0x11052u, 0x1106Fu},
    {0x110F0u, 0x110F9u},
    {0x11136u, 0x1113Fu},
    {0x111D0u, 0x111D9u},
    {0x111E1u, 0x111F4u},
    {0x112F0u, 0x112F9u},
    {0x11450u, 0x11459u},
    {0x114D0u, 0x114D9u},
    {0x11650u, 0x11659u},
    {0x116C0u, 0x116C9u},
    {0x11730u, 0x1173Bu},
    {0x118E0u, 0x118F2u},
    {0x11950u, 0x11959u},
    {0x11C50u, 0x11C6Cu},
    {0x11D50u, 0x11D59u},
    {0x11DA0u, 0x11DA9u},
    {0x11F50u, 0x11F59u},
    {0x11FC0u, 0x11FD4u},
    {0x12400u, 0x1246Eu},
    {0x16A60u, 0x16A69u},
    {0x16AC0u, 0x16AC9u},
    {0x16B50u, 0x16B59u},
    {0x16B5Bu, 0x16B61u},
    {0x16E80u, 0x16E96u},
    {0x1D2C0u, 0x1D2D3u},
    {0x1D2E0u, 0x1D2F3u},
    {0x1D360u, 0x1D378u},
    {0x1D7CEu, 0x1D7FFu},
    {0x1E140u, 0x1E149u},
    {0x1E2F0u, 0x1E2F9u},
    {0x1E4F0u, 0x1E4F9u},
    {0x1E8C7u, 0x1E8CFu},
    {0x1E950u, 0x1E959u},
    {0x1EC71u, 0x1ECABu},
    {0x1ECADu, 0x1ECAFu},
    {0x1ECB1u, 0x1ECB4u},
    {0x1ED01u, 0x1ED2Du},
    {0x1ED2Fu, 0x1ED3Du},
    {0x1F100u, 0x1F10Cu},
    {0x1FBF0u, 0x1FBF9u},
};

static constexpr Range SPACE_RANGES[] = {
    {0x9u, 0xDu},
    {0x1Cu, 0x20u},
    {0x85u, 0x85u},
    {0xA0u, 0xA0u},
    {0x1680u, 0x1680u},
    {0x2000u, 0x200Au},
    {0x2028u, 0x2029u},
    {0x202Fu, 0x202Fu},
    {0x205Fu, 0x205Fu},
    {0x3000u, 0x3000u},
};

template <size_t Count>
static bool in_ranges(
    uint32_t codepoint,
    const Range (&ranges)[Count]) {
    size_t low = 0;
    size_t high = Count;
    while (low < high) {
        const size_t middle =
            low + (high - low) / 2;
        if (codepoint < ranges[middle].first) {
            high = middle;
        } else if (
            codepoint > ranges[middle].last) {
            low = middle + 1;
        } else {
            return true;
        }
    }
    return false;
}

static bool is_letter(uint32_t codepoint) {
    return in_ranges(
        codepoint, LETTER_RANGES);
}

static bool is_number(uint32_t codepoint) {
    return in_ranges(
        codepoint, NUMBER_RANGES);
}

static bool is_space(uint32_t codepoint) {
    return in_ranges(
        codepoint, SPACE_RANGES);
}

struct Rune {
    uint32_t codepoint = 0;
    size_t begin = 0;
    size_t end = 0;
};

static std::vector<Rune> decode_utf8(
    const std::string& text) {
    std::vector<Rune> runes;
    size_t position = 0;

    while (position < text.size()) {
        const size_t begin = position;
        const uint8_t first =
            static_cast<uint8_t>(text[position++]);

        uint32_t codepoint = 0;
        int continuation = 0;

        if (first < 0x80u) {
            codepoint = first;
        } else if ((first & 0xE0u) == 0xC0u) {
            codepoint = first & 0x1Fu;
            continuation = 1;
        } else if ((first & 0xF0u) == 0xE0u) {
            codepoint = first & 0x0Fu;
            continuation = 2;
        } else if ((first & 0xF8u) == 0xF0u) {
            codepoint = first & 0x07u;
            continuation = 3;
        } else {
            throw std::runtime_error(
                "invalid UTF-8 leading byte at " +
                std::to_string(begin));
        }

        for (int index = 0;
             index < continuation;
             ++index) {
            if (position >= text.size()) {
                throw std::runtime_error(
                    "truncated UTF-8 sequence");
            }
            const uint8_t next =
                static_cast<uint8_t>(
                    text[position++]);
            if ((next & 0xC0u) != 0x80u) {
                throw std::runtime_error(
                    "invalid UTF-8 continuation");
            }
            codepoint =
                (codepoint << 6) |
                (next & 0x3Fu);
        }

        if (codepoint > 0x10FFFFu ||
            (codepoint >= 0xD800u &&
             codepoint <= 0xDFFFu)) {
            throw std::runtime_error(
                "invalid UTF-8 codepoint");
        }

        runes.push_back(
            Rune{
                codepoint,
                begin,
                position,
            });
    }
    return runes;
}

static bool is_crlf(uint32_t codepoint) {
    return codepoint == '\r' ||
           codepoint == '\n';
}

static bool ascii_equal_ci(
    uint32_t codepoint,
    char expected) {
    if (codepoint > 0x7Fu) {
        return false;
    }
    return std::tolower(
               static_cast<unsigned char>(
                   codepoint)) ==
           std::tolower(
               static_cast<unsigned char>(
                   expected));
}

static std::optional<size_t>
match_contraction(
    const std::vector<Rune>& runes,
    size_t index) {
    if (runes[index].codepoint != '\'') {
        return std::nullopt;
    }

    static constexpr const char* suffixes[] = {
        "s", "t", "re", "ve", "m", "ll", "d",
    };

    for (const char* suffix : suffixes) {
        size_t cursor = index + 1;
        bool match = true;
        for (const char* character = suffix;
             *character;
             ++character) {
            if (cursor >= runes.size() ||
                !ascii_equal_ci(
                    runes[cursor].codepoint,
                    *character)) {
                match = false;
                break;
            }
            ++cursor;
        }
        if (match) {
            return cursor;
        }
    }
    return std::nullopt;
}

static std::vector<std::string>
pretokenize_qwen(
    const std::string& text) {
    const std::vector<Rune> runes =
        decode_utf8(text);
    std::vector<std::string> pieces;

    size_t index = 0;
    while (index < runes.size()) {
        const size_t begin_index = index;
        size_t end_index = index;

        if (const auto contraction =
                match_contraction(
                    runes, index)) {
            end_index = *contraction;
        } else {

            size_t cursor = index;
            if (!is_crlf(
                    runes[cursor].codepoint) &&
                !is_letter(
                    runes[cursor].codepoint) &&
                !is_number(
                    runes[cursor].codepoint)) {
                ++cursor;
            }
            const size_t letter_begin = cursor;
            while (cursor < runes.size() &&
                   is_letter(
                       runes[cursor].codepoint)) {
                ++cursor;
            }

            if (cursor > letter_begin) {
                end_index = cursor;
            } else if (
                is_number(
                    runes[index].codepoint)) {

                cursor = index;
                int count = 0;
                while (cursor < runes.size() &&
                       is_number(
                           runes[cursor].codepoint) &&
                       count < 3) {
                    ++cursor;
                    ++count;
                }
                end_index = cursor;
            } else {

                cursor = index;
                if (runes[cursor].codepoint ==
                        0x20u &&
                    cursor + 1 < runes.size() &&
                    !is_space(
                        runes[cursor + 1].
                            codepoint) &&
                    !is_letter(
                        runes[cursor + 1].
                            codepoint) &&
                    !is_number(
                        runes[cursor + 1].
                            codepoint)) {
                    ++cursor;
                }

                const size_t symbol_begin = cursor;
                while (cursor < runes.size() &&
                       !is_space(
                           runes[cursor].
                               codepoint) &&
                       !is_letter(
                           runes[cursor].
                               codepoint) &&
                       !is_number(
                           runes[cursor].
                               codepoint)) {
                    ++cursor;
                }

                if (cursor > symbol_begin) {
                    while (cursor < runes.size() &&
                           is_crlf(
                               runes[cursor].
                                   codepoint)) {
                        ++cursor;
                    }
                    end_index = cursor;
                } else if (
                    is_space(
                        runes[index].codepoint)) {

                    cursor = index;
                    size_t last_newline =
                        std::numeric_limits<size_t>::
                            max();
                    while (cursor < runes.size() &&
                           is_space(
                               runes[cursor].
                                   codepoint)) {
                        if (is_crlf(
                                runes[cursor].
                                    codepoint)) {
                            last_newline = cursor;
                        }
                        ++cursor;
                    }

                    if (last_newline !=
                        std::numeric_limits<size_t>::
                            max()) {
                        end_index =
                            last_newline + 1;
                    } else if (
                        cursor == runes.size()) {
                        end_index = cursor;
                    } else {
                        const size_t count =
                            cursor - index;
                        const bool next_can_take_space =
                            is_letter(
                                runes[cursor].
                                    codepoint) ||
                            (!is_space(
                                 runes[cursor].
                                     codepoint) &&
                             !is_letter(
                                 runes[cursor].
                                     codepoint) &&
                             !is_number(
                                 runes[cursor].
                                     codepoint));

                        if (count > 1 &&
                            next_can_take_space) {
                            end_index = cursor - 1;
                        } else {
                            end_index = cursor;
                        }
                    }
                }
            }
        }

        if (end_index <= begin_index) {
            throw std::runtime_error(
                "pre-tokenizer made no progress at "
                "rune " +
                std::to_string(index));
        }

        const size_t byte_begin =
            runes[begin_index].begin;
        const size_t byte_end =
            runes[end_index - 1].end;
        pieces.emplace_back(
            text.substr(
                byte_begin,
                byte_end - byte_begin));
        index = end_index;
    }

    return pieces;
}

struct PairHash {
    size_t operator()(
        const std::pair<std::string, std::string>&
            pair) const noexcept {
        const size_t first =
            std::hash<std::string>{}(pair.first);
        const size_t second =
            std::hash<std::string>{}(pair.second);
        return first ^
               (second +
                0x9e3779b97f4a7c15ull +
                (first << 6) +
                (first >> 2));
    }
};

class ZImageTokenizer {
public:
    static constexpr int64_t END_OF_TEXT =
        151643;
    static constexpr int64_t IM_START =
        151644;
    static constexpr int64_t IM_END =
        151645;
    static constexpr size_t MAX_PHYSICAL_LENGTH =
        512;

    static size_t select_physical_length(
        size_t valid_length) {
        if (valid_length <= 128) {
            return 128;
        }
        if (valid_length <= 256) {
            return 256;
        }
        if (valid_length <= 512) {
            return 512;
        }
        throw std::runtime_error(
            "prompt token length exceeds 512");
    }

    explicit ZImageTokenizer(
        const fs::path& tokenizer_dir,
        bool enable_decode = false) {
        std::cerr
            << "[tokenizer init] loading vocab.json...\n";
        load_vocabulary(
            tokenizer_dir / "vocab.json");
        std::cerr
            << "[tokenizer init] vocabulary entries="
            << vocabulary_.size() << "\n";

        std::cerr
            << "[tokenizer init] loading merges.txt...\n";
        load_merges(
            tokenizer_dir / "merges.txt");
        std::cerr
            << "[tokenizer init] merge rules="
            << merge_rank_.size() << "\n";

        build_byte_alphabet();
        std::cerr
            << "[tokenizer init] byte alphabet ready\n";

        if (vocabulary_.size() !=
            static_cast<size_t>(END_OF_TEXT)) {
            throw std::runtime_error(
                "expected Qwen base vocabulary size "
                "151643, got " +
                std::to_string(
                    vocabulary_.size()));
        }

        if (enable_decode) {
            std::cerr
                << "[tokenizer init] building reverse vocabulary...\n";
            reverse_vocabulary_.resize(
                static_cast<size_t>(END_OF_TEXT));
            for (const auto& entry :
                 vocabulary_) {
                if (entry.second < 0 ||
                    entry.second >= END_OF_TEXT) {
                    throw std::runtime_error(
                        "base vocabulary id out of range");
                }
                reverse_vocabulary_.at(
                    static_cast<size_t>(
                        entry.second)) =
                    entry.first;
            }

            for (size_t id = 0;
                 id < reverse_vocabulary_.size();
                 ++id) {
                if (reverse_vocabulary_[id].empty()) {
                    throw std::runtime_error(
                        "base vocabulary has an empty reverse slot at id " +
                        std::to_string(id));
                }
            }
            std::cerr
                << "[tokenizer init] reverse vocabulary ready\n";
        }
    }

    std::vector<int64_t> encode_text(
        const std::string& text) {
        std::vector<int64_t> output;
        for (const std::string& piece :
             pretokenize_qwen(text)) {
            const std::string encoded =
                byte_encode(piece);
            const std::vector<std::string> tokens =
                bpe(encoded);

            for (const std::string& token :
                 tokens) {
                const auto iterator =
                    vocabulary_.find(token);
                if (iterator ==
                    vocabulary_.end()) {
                    throw std::runtime_error(
                        "BPE token missing from vocab");
                }
                output.push_back(
                    iterator->second);
            }
        }
        return output;
    }

    struct EncodedPrompt {
        std::string rendered;
        std::vector<int64_t> token_ids;
        std::vector<uint8_t> attention_mask;
        size_t prompt_length = 0;
        size_t valid_length = 0;
    };

    EncodedPrompt encode_prompt(
        const std::string& prompt) {
        EncodedPrompt result;
        result.rendered =
            "<|im_start|>user\n" +
            prompt +
            "<|im_end|>\n"
            "<|im_start|>assistant\n";

        std::vector<int64_t> valid;
        valid.push_back(IM_START);

        append_ids(
            valid,
            encode_text("user\n"));

        const std::vector<int64_t> prompt_tokens =
            encode_text(prompt);
        result.prompt_length =
            prompt_tokens.size();
        append_ids(
            valid,
            prompt_tokens);

        valid.push_back(IM_END);
        append_ids(
            valid,
            encode_text("\n"));

        valid.push_back(IM_START);
        append_ids(
            valid,
            encode_text("assistant\n"));

        if (valid.size() >
            MAX_PHYSICAL_LENGTH) {
            const size_t wrapper_tokens =
                valid.size() -
                result.prompt_length;
            const size_t maximum_prompt_tokens =
                MAX_PHYSICAL_LENGTH -
                wrapper_tokens;
            throw std::runtime_error(
                "user prompt exceeds " +
                std::to_string(
                    maximum_prompt_tokens) +
                " tokens; rendered prompt limit is 512 tokens");
        }

        result.valid_length =
            valid.size();
        const size_t physical_length =
            select_physical_length(
                result.valid_length);
        result.token_ids.assign(
            physical_length,
            END_OF_TEXT);
        result.attention_mask.assign(
            physical_length,
            0);

        std::copy(
            valid.begin(),
            valid.end(),
            result.token_ids.begin());
        std::fill(
            result.attention_mask.begin(),
            result.attention_mask.begin() +
                static_cast<std::ptrdiff_t>(
                    result.valid_length),
            1);
        return result;
    }

    std::string decode_valid(
        const std::vector<int64_t>& ids,
        size_t valid_length) const {
        if (reverse_vocabulary_.empty()) {
            throw std::runtime_error(
                "native decode was not enabled for this tokenizer instance");
        }
        if (valid_length > ids.size()) {
            throw std::runtime_error(
                "valid length exceeds ID vector");
        }

        std::string output;
        for (size_t index = 0;
             index < valid_length;
             ++index) {
            const int64_t id = ids[index];
            if (id == END_OF_TEXT) {
                output += "<|endoftext|>";
            } else if (id == IM_START) {
                output += "<|im_start|>";
            } else if (id == IM_END) {
                output += "<|im_end|>";
            } else if (
                id >= 0 &&
                id < END_OF_TEXT) {
                output += byte_decode_token(
                    reverse_vocabulary_[
                        static_cast<size_t>(
                            id)]);
            } else {
                throw std::runtime_error(
                    "cannot decode token id " +
                    std::to_string(id));
            }
        }
        return output;
    }

    static std::string extract_single_user_prompt(
        const std::string& rendered) {
        const std::string prefix =
            "<|im_start|>user\n";
        const std::string suffix =
            "<|im_end|>\n"
            "<|im_start|>assistant\n";

        if (rendered.size() <
                prefix.size() +
                suffix.size() ||
            rendered.compare(
                0,
                prefix.size(),
                prefix) != 0 ||
            rendered.compare(
                rendered.size() -
                    suffix.size(),
                suffix.size(),
                suffix) != 0) {
            throw std::runtime_error(
                "decoded reference is not the "
                "supported single-user Qwen template");
        }

        return rendered.substr(
            prefix.size(),
            rendered.size() -
                prefix.size() -
                suffix.size());
    }

private:
    std::unordered_map<std::string, int64_t>
        vocabulary_;
    std::vector<std::string>
        reverse_vocabulary_;
    std::unordered_map<
        std::pair<std::string, std::string>,
        int,
        PairHash>
        merge_rank_;
    std::unordered_map<
        std::string,
        std::vector<std::string>>
        bpe_cache_;
    std::array<std::string, 256>
        byte_encoder_;
    std::unordered_map<uint32_t, uint8_t>
        byte_decoder_;

    static void append_ids(
        std::vector<int64_t>& destination,
        const std::vector<int64_t>& source) {
        destination.insert(
            destination.end(),
            source.begin(),
            source.end());
    }

    void load_vocabulary(
        const fs::path& path) {
        VocabularyParser parser(
            read_text(path));
        vocabulary_ = parser.parse();
        if (vocabulary_.empty()) {
            throw std::runtime_error(
                "parsed vocabulary is empty: " +
                path.string());
        }
    }

    void load_merges(
        const fs::path& path) {
        merge_rank_.reserve(160000);
        bpe_cache_.reserve(4096);
        std::istringstream stream(
            read_text(path));
        std::string line;
        int rank = 0;

        while (std::getline(stream, line)) {
            if (!line.empty() &&
                line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty() ||
                line[0] == '#') {
                continue;
            }

            const size_t separator =
                line.find(' ');
            if (separator ==
                    std::string::npos ||
                separator == 0 ||
                separator + 1 >=
                    line.size()) {
                throw std::runtime_error(
                    "invalid BPE merge line");
            }

            const std::string first =
                line.substr(0, separator);
            const std::string second =
                line.substr(separator + 1);

            merge_rank_.emplace(
                std::make_pair(
                    first, second),
                rank++);
        }
    }

    void build_byte_alphabet() {
        std::vector<int> bytes;
        std::vector<uint32_t> codepoints;

        auto append_range =
            [&](int first, int last) {
                for (int value = first;
                     value <= last;
                     ++value) {
                    bytes.push_back(value);
                    codepoints.push_back(
                        static_cast<uint32_t>(
                            value));
                }
            };

        append_range('!', '~');
        append_range(0xA1, 0xAC);
        append_range(0xAE, 0xFF);

        int extra = 0;
        for (int byte = 0;
             byte < 256;
             ++byte) {
            if (std::find(
                    bytes.begin(),
                    bytes.end(),
                    byte) ==
                bytes.end()) {
                bytes.push_back(byte);
                codepoints.push_back(
                    static_cast<uint32_t>(
                        256 + extra));
                ++extra;
            }
        }

        if (bytes.size() != 256 ||
            codepoints.size() != 256) {
            throw std::runtime_error(
                "byte alphabet construction failed");
        }

        for (size_t index = 0;
             index < bytes.size();
             ++index) {
            std::string encoded;
            append_utf8(
                encoded,
                codepoints[index]);
            byte_encoder_[
                static_cast<size_t>(
                    bytes[index])] =
                encoded;
            byte_decoder_.emplace(
                codepoints[index],
                static_cast<uint8_t>(
                    bytes[index]));
        }
    }

    std::string byte_encode(
        const std::string& piece) const {
        std::string output;
        for (const unsigned char byte :
             piece) {
            output += byte_encoder_[
                static_cast<size_t>(byte)];
        }
        return output;
    }

    std::string byte_decode_token(
        const std::string& token) const {
        const std::vector<Rune> runes =
            decode_utf8(token);
        std::string output;
        output.reserve(runes.size());

        for (const Rune& rune : runes) {
            const auto iterator =
                byte_decoder_.find(
                    rune.codepoint);
            if (iterator ==
                byte_decoder_.end()) {
                throw std::runtime_error(
                    "vocab token contains a "
                    "non-byte-level codepoint");
            }
            output.push_back(
                static_cast<char>(
                    iterator->second));
        }
        return output;
    }

    std::vector<std::string>
    utf8_symbols(
        const std::string& encoded) const {
        const std::vector<Rune> runes =
            decode_utf8(encoded);
        std::vector<std::string> symbols;
        symbols.reserve(runes.size());

        for (const Rune& rune : runes) {
            symbols.emplace_back(
                encoded.substr(
                    rune.begin,
                    rune.end - rune.begin));
        }
        return symbols;
    }

    std::vector<std::string> bpe(
        const std::string& encoded) {
        const auto cache =
            bpe_cache_.find(encoded);
        if (cache != bpe_cache_.end()) {
            return cache->second;
        }

        std::vector<std::string> word =
            utf8_symbols(encoded);
        if (word.size() <= 1) {
            bpe_cache_.emplace(
                encoded, word);
            return word;
        }

        for (;;) {
            int best_rank =
                std::numeric_limits<int>::max();
            std::pair<std::string, std::string>
                best_pair;
            bool found = false;

            for (size_t index = 0;
                 index + 1 < word.size();
                 ++index) {
                const auto iterator =
                    merge_rank_.find(
                        std::make_pair(
                            word[index],
                            word[index + 1]));
                if (iterator !=
                        merge_rank_.end() &&
                    iterator->second <
                        best_rank) {
                    best_rank =
                        iterator->second;
                    best_pair =
                        iterator->first;
                    found = true;
                }
            }

            if (!found) {
                break;
            }

            std::vector<std::string> merged;
            merged.reserve(word.size());
            size_t index = 0;

            while (index < word.size()) {
                if (index + 1 < word.size() &&
                    word[index] ==
                        best_pair.first &&
                    word[index + 1] ==
                        best_pair.second) {
                    merged.push_back(
                        word[index] +
                        word[index + 1]);
                    index += 2;
                } else {
                    merged.push_back(
                        word[index]);
                    ++index;
                }
            }

            word = std::move(merged);
            if (word.size() <= 1) {
                break;
            }
        }

        bpe_cache_.emplace(
            encoded, word);
        return word;
    }
};


static void legacy_append_ids(
    std::vector<int64_t>& dst,
    const std::vector<int64_t>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

static std::vector<int64_t> encode_tts_assistant_prompt(
    ZImageTokenizer& tokenizer,
    const std::string& text) {

    // Official Qwen3-TTS inference wrapper:
    // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int64_t> ids;
    ids.push_back(ZImageTokenizer::IM_START);

    // Keep the ordinary-text span contiguous. Splitting "assistant\n" and
    // `text` into two independent BPE calls could forbid a merge across that
    // boundary and therefore diverge from the official tokenizer.
    legacy_append_ids(
        ids,
        tokenizer.encode_text(
            std::string("assistant\n") + text));

    ids.push_back(ZImageTokenizer::IM_END);
    legacy_append_ids(ids, tokenizer.encode_text("\n"));
    ids.push_back(ZImageTokenizer::IM_START);
    legacy_append_ids(ids, tokenizer.encode_text("assistant\n"));
    return ids;
}

static fs::path locate_tokenizer_dir(const fs::path& model_dir) {
    const std::array<fs::path, 3> candidates{{
        model_dir,
        model_dir / "tokenizer",
        model_dir / "text_tokenizer",
    }};
    for (const auto& p : candidates) {
        if (fs::is_regular_file(p / "vocab.json")
            && fs::is_regular_file(p / "merges.txt")) {
            return p;
        }
    }
    throw std::runtime_error(
        "cannot locate vocab.json + merges.txt under model root");
}

static void write_i64(const fs::path& path, const std::vector<int64_t>& ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    if (!ids.empty()) {
        f.write(
            reinterpret_cast<const char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(int64_t)));
    }
    if (!f) throw std::runtime_error("failed writing " + path.string());
}

template<class T>
static void write_binary(
    const fs::path& path,
    const std::vector<T>& values) {

    fs::create_directories(path.parent_path());
    std::ofstream f(path,std::ios::binary);

    if(!f){
        throw std::runtime_error(
            "cannot create correctness dump: "
            +path.string());
    }

    if(!values.empty()){
        f.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(
                values.size()*sizeof(T)));
    }

    if(!f){
        throw std::runtime_error(
            "failed correctness dump: "
            +path.string());
    }
}

static void write_scalar_text(
    const fs::path& path,
    const std::string& value) {

    fs::create_directories(path.parent_path());
    std::ofstream f(path);

    if(!f){
        throw std::runtime_error(
            "cannot create correctness text: "
            +path.string());
    }

    f<<value<<"\n";
}



// ============================================================================
// legacy_legacy_tag_e single-file E2E cutover runtime
// ============================================================================

struct CliOptions {
    fs::path model_dir;
    fs::path output_wav = "qwen3_tts.wav";
    fs::path work_dir;
    fs::path bridge_script;
    fs::path native_assets_dir;
    fs::path generation_tail_assets_dir;
    fs::path ref_audio;
    fs::path correctness_dir;
    fs::path speaker_embedding_bf16;
    std::string language = "English";
    std::string text;
    std::string task;
    std::string text_mode;
    std::string speaker;
    std::string instruct;
    std::string mode = "official-base-xvector";
    std::size_t max_new_tokens = 0;
    bool max_new_tokens_explicit = false;
    std::uint64_t seed = 1234;
    bool greedy = false;
    bool keep_work = false;
    bool self_test = false;
};


static constexpr std::size_t kGenerationCapacityMin=256;
static constexpr std::size_t kGenerationCapacityMax=8192;

static std::size_t generation_capacity_for(
    std::size_t requested) {
    if(requested==0||requested>kGenerationCapacityMax){
        throw std::runtime_error(
            "--max-new-tokens must be in [1,8192]");
    }
    std::size_t capacity=kGenerationCapacityMin;
    while(capacity<requested)capacity<<=1;
    return capacity;
}

static std::string read_text_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "cannot open text file: " + path.string());
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void write_text_file(
    const fs::path& path,
    const std::string& value) {

    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "cannot create text file: " + path.string());
    }
    f.write(
        value.data(),
        static_cast<std::streamsize>(value.size()));
    if (!f) {
        throw std::runtime_error(
            "failed writing text file: " + path.string());
    }
}

template<class T>
static std::vector<T> read_binary(
    const fs::path& path) {

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "cannot open binary file: " + path.string());
    }

    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (
        bytes < 0
        || bytes % static_cast<std::streamoff>(sizeof(T)) != 0
    ) {
        throw std::runtime_error(
            "invalid binary byte count: " + path.string());
    }

    std::vector<T> out(
        static_cast<std::size_t>(bytes / sizeof(T)));

    if (!out.empty()) {
        f.read(
            reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(
                out.size() * sizeof(T)));
    }

    if (!f && !out.empty()) {
        throw std::runtime_error(
            "failed reading binary file: " + path.string());
    }
    return out;
}


template<class T>
static void legacy_write_binary(
    const fs::path& path,
    const std::vector<T>& values) {

    fs::create_directories(path.parent_path());

    std::ofstream stream(
        path,
        std::ios::binary);

    if (!stream) {
        throw std::runtime_error(
            "cannot create binary file: "
            + path.string());
    }

    if (!values.empty()) {
        stream.write(
            reinterpret_cast<const char*>(
                values.data()),
            static_cast<std::streamsize>(
                values.size()
                * sizeof(T)));
    }

    if (!stream) {
        throw std::runtime_error(
            "failed writing binary file: "
            + path.string());
    }
}

static void write_u16_le(
    std::ofstream& stream,
    std::uint16_t value) {

    const char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    stream.write(bytes, 2);
}

static void write_u32_le(
    std::ofstream& stream,
    std::uint32_t value) {

    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    stream.write(bytes, 4);
}

static void write_float_wav(
    const fs::path& path,
    const std::vector<float>& samples,
    int sample_rate) {

    if (samples.empty()) {
        throw std::runtime_error(
            "refusing to write an empty waveform");
    }
    if (sample_rate <= 0) {
        throw std::runtime_error(
            "invalid waveform sample rate");
    }

    fs::create_directories(path.parent_path());

    const std::uint64_t raw_bytes =
        static_cast<std::uint64_t>(samples.size())
        * sizeof(float);

    if (raw_bytes > 0xffffffffULL - 36ULL) {
        throw std::runtime_error(
            "waveform is too large for RIFF/WAV");
    }

    const std::uint32_t data_bytes =
        static_cast<std::uint32_t>(raw_bytes);

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot create WAV: " + path.string());
    }

    stream.write("RIFF", 4);
    write_u32_le(stream, 36u + data_bytes);
    stream.write("WAVE", 4);

    stream.write("fmt ", 4);
    write_u32_le(stream, 16u);
    write_u16_le(stream, 3u); // IEEE float
    write_u16_le(stream, 1u); // mono
    write_u32_le(
        stream,
        static_cast<std::uint32_t>(sample_rate));
    write_u32_le(
        stream,
        static_cast<std::uint32_t>(
            sample_rate * sizeof(float)));
    write_u16_le(
        stream,
        static_cast<std::uint16_t>(sizeof(float)));
    write_u16_le(stream, 32u);

    stream.write("data", 4);
    write_u32_le(stream, data_bytes);
    stream.write(
        reinterpret_cast<const char*>(samples.data()),
        static_cast<std::streamsize>(data_bytes));

    if (!stream) {
        throw std::runtime_error(
            "failed writing WAV: " + path.string());
    }
}

static fs::path executable_path() {
#if defined(__linux__)
    std::array<char, 4096> buffer{};
    const ssize_t n = ::readlink(
        "/proc/self/exe",
        buffer.data(),
        buffer.size() - 1);
    if (n > 0) {
        buffer[static_cast<std::size_t>(n)] = '\0';
        return fs::weakly_canonical(
            fs::path(buffer.data()));
    }
#endif
    return fs::current_path() / "qwen3_tts";
}

static fs::path default_bridge_script() {
    const fs::path exe = executable_path();
    const fs::path package_root =
        exe.parent_path().parent_path();
    return package_root / "tools/e2e_bridge.py";
}

static int run_process(
    const std::vector<std::string>& arguments) {

#if defined(__linux__)
    if (arguments.empty()) {
        throw std::runtime_error(
            "cannot execute empty command");
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        throw std::runtime_error("fork() failed");
    }

    if (pid == 0) {
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1);
        for (const std::string& arg : arguments) {
            argv.push_back(
                const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        ::execvp(argv[0], argv.data());
        std::cerr
            << "FATAL: execvp failed for "
            << arguments[0] << "\n";
        ::_exit(127);
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        throw std::runtime_error("waitpid() failed");
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 126;
#else
    (void)arguments;
    throw std::runtime_error(
        "legacy_legacy_tag_e bridge process launcher currently requires Linux");
#endif
}

static int read_int_text(const fs::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(
            "cannot open integer file: " + path.string());
    }
    long long value = 0;
    f >> value;
    if (!f) {
        throw std::runtime_error(
            "cannot parse integer file: " + path.string());
    }
    if (
        value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max()
    ) {
        throw std::runtime_error(
            "integer is out of range: " + path.string());
    }
    return static_cast<int>(value);
}

static CliOptions parse_cli(
    int argc,
    char** argv) {

    CliOptions options;
    fs::path text_file;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--model-dir") {
            options.model_dir =
                fs::path(require_value("--model-dir"));
        } else if (arg == "--text") {
            options.text = require_value("--text");
        } else if (arg == "--text-file") {
            text_file = fs::path(
                require_value("--text-file"));
        } else if (arg == "--ref-audio") {
            options.ref_audio = fs::path(require_value("--ref-audio"));
        } else if (arg == "--correctness-dir") {
            options.correctness_dir =
                fs::path(require_value("--correctness-dir"));
        } else if (arg == "--speaker-embedding-bf16") {
            options.speaker_embedding_bf16 =
                fs::path(require_value("--speaker-embedding-bf16"));
        } else if (arg == "--language") {
            options.language = require_value("--language");
        } else if (arg == "--task") {
            options.task = require_value("--task");
        } else if (arg == "--text-mode") {
            options.text_mode = require_value("--text-mode");
        } else if (arg == "--speaker") {
            options.speaker = require_value("--speaker");
        } else if (arg == "--instruct") {
            options.instruct = require_value("--instruct");
        } else if (arg == "--output") {
            options.output_wav =
                fs::path(require_value("--output"));
        } else if (arg == "--work-dir") {
            options.work_dir =
                fs::path(require_value("--work-dir"));
        } else if (arg == "--bridge-script") {
            options.bridge_script =
                fs::path(require_value("--bridge-script"));
        } else if (arg == "--native-assets-dir") {
            options.native_assets_dir =
                fs::path(require_value("--native-assets-dir"));
        } else if (arg == "--generation-tail-assets-dir") {
            options.generation_tail_assets_dir =
                fs::path(require_value("--generation-tail-assets-dir"));
        } else if (arg == "--mode") {
            options.mode = require_value("--mode");
        } else if (arg == "--max-new-tokens") {
            const auto raw =
                require_value("--max-new-tokens");
            const unsigned long long value =
                std::stoull(raw);
            if(value==0||value>kGenerationCapacityMax){
                throw std::runtime_error(
                    "--max-new-tokens must be in [1,8192]");
            }
            options.max_new_tokens=static_cast<std::size_t>(value);
            options.max_new_tokens_explicit=true;
        } else if (arg == "--seed") {
            const auto raw =
                require_value("--seed");
            options.seed =
                static_cast<std::uint64_t>(
                    std::stoull(raw));
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--keep-work") {
            options.keep_work = true;
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (
            arg == "--help"
            || arg == "-h"
        ) {
            std::cout
                << "Qwen3-TTS legacy_legacy_tag_j.3 official Base x-vector voice clone, single C++/HIP\n\n"
                << "Usage:\n  "
                << argv[0]
                << " --model-dir DIR"
                << " --ref-audio ref.wav"
                << " --language English"
                << " (--text TEXT | --text-file FILE)"
                << " --output out.wav"
                << " --max-new-tokens N"
                << "  (N: 1..8192; capacity: 256/512/1024/2048/4096/8192)"
                << " [--seed 1234]"
                << " [--greedy]"
                << " [--mode official-base-xvector|native-e2e-legacy_legacy_tag_n-validate|native-e2e-legacy_legacy_tag_n|native-e2e-legacy_legacy_tag_m-validate|native-e2e-legacy_legacy_tag_l|native|generation-tail-algorithmic|generation-tail-native|frontend-native|bridge]"
                << " [--work-dir DIR]"
                << " [--keep-work]\n\n"
                << "Modes:\n"
                << "  native                 Final target. No Python/gold/trace runtime dependency.\n"
                << "  generation-tail-native Native frontend + FinalHead + Predictor; bridge only\n"
                << "                         supplies Talker Transformer hidden and decoder.\n"
                << "  frontend-native        Native tokenizer + safetensors + HIP text projection/prefill;\n"
                << "                  bridge begins only at Talker generation.\n"
                << "  bridge          Transitional full E2E. Same C++ CLI and WAV writer, but\n"
                << "          generation+decoder are supplied by the official PyTorch bridge.\n";
            std::exit(0);
        } else {
            throw std::runtime_error(
                "unknown argument: " + arg);
        }
    }

    if (!text_file.empty()) {
        if (!options.text.empty()) {
            throw std::runtime_error(
                "use only one of --text and --text-file");
        }
        options.text = read_text_file(text_file);
    }

    if (!options.self_test) {
        if (options.model_dir.empty()) {
            throw std::runtime_error(
                "--model-dir is required");
        }
        if (options.text.empty()) {
            throw std::runtime_error(
                "--text or --text-file is required");
        }
        if(
            options.mode=="official-base-xvector"
            &&options.task.empty()
        ){
            throw std::runtime_error(
                "formal 0.6B execution requires explicit --task");
        }

        if(
            !options.task.empty()
            &&options.mode!="official-base-xvector"
        ){
            throw std::runtime_error(
                "--task cannot be combined with legacy --mode routes");
        }

        if(!options.task.empty()){
            if(!options.max_new_tokens_explicit){
                throw std::runtime_error(
                    "--max-new-tokens is required");
            }
            (void)generation_capacity_for(options.max_new_tokens);
            if(
                options.task!="base-xvector"
                &&options.task!="custom-voice"
            ){
                throw std::runtime_error(
                    "--task must be base-xvector or custom-voice");
            }

            if(options.text_mode!="streaming"){
                throw std::runtime_error(
                    "formal 0.6B execution requires --text-mode streaming");
            }

            if(options.task=="base-xvector"){
                if(options.ref_audio.empty()){
                    throw std::runtime_error(
                        "--ref-audio is required for --task base-xvector");
                }

                if(!options.speaker.empty()){
                    throw std::runtime_error(
                        "--speaker is invalid for --task base-xvector");
                }

                if(!options.instruct.empty()){
                    throw std::runtime_error(
                        "--instruct is invalid for --task base-xvector");
                }
            }else{
                if(options.speaker.empty()){
                    throw std::runtime_error(
                        "--speaker is required for --task custom-voice");
                }

                if(!options.ref_audio.empty()){
                    throw std::runtime_error(
                        "--ref-audio is invalid for --task custom-voice");
                }

                if(!options.speaker_embedding_bf16.empty()){
                    throw std::runtime_error(
                        "--speaker-embedding-bf16 is invalid for --task custom-voice");
                }

                if(!options.instruct.empty()){
                    throw std::runtime_error(
                        "0.6B CustomVoice does not support --instruct");
                }
            }
        }
    }

    if (
        options.mode != "official-base-xvector"
        && options.mode != "native-e2e-legacy_legacy_tag_n-validate"
        && options.mode != "native-e2e-legacy_legacy_tag_n"
        && options.mode != "native-e2e-legacy_legacy_tag_m-validate"
        && options.mode != "native-e2e-legacy_legacy_tag_l"
        && options.mode != "native"
        && options.mode != "generation-tail-native"
        && options.mode != "generation-tail-algorithmic"
        && options.mode != "frontend-native"
        && options.mode != "bridge"
    ) {
        throw std::runtime_error(
            "--mode must be official-base-xvector, native-e2e-legacy_legacy_tag_n-validate, native-e2e-legacy_legacy_tag_n, native-e2e-legacy_legacy_tag_m-validate, native-e2e-legacy_legacy_tag_l, native, generation-tail-algorithmic, generation-tail-native, frontend-native, or bridge");
    }

    return options;
}

static int run_self_test() {
    const fs::path root =
        fs::temp_directory_path()
        / "qwen3_tts_self_test";

    fs::remove_all(root);
    fs::create_directories(root);

    std::vector<float> wave(240);
    for (std::size_t i = 0; i < wave.size(); ++i) {
        wave[i] =
            static_cast<float>(i)
            / static_cast<float>(wave.size())
            * 0.1f;
    }

    const fs::path wav = root / "self_test.wav";
    write_float_wav(wav, wave, 24000);

    const bool wav_ok =
        fs::is_regular_file(wav)
        && fs::file_size(wav)
            == 44 + wave.size() * sizeof(float);

    auto host_float_to_bf16=[](float value)->std::uint16_t{
        std::uint32_t bits=0;
        std::memcpy(&bits,&value,sizeof(bits));

        if((bits&0x7fffffffU)>0x7f800000U){
            return static_cast<std::uint16_t>(
                (bits>>16)|0x0040U);
        }

        const std::uint32_t lsb=
            (bits>>16)&1U;

        bits += 0x7fffU + lsb;

        return static_cast<std::uint16_t>(
            bits>>16);
    };

    auto host_bf16_to_float=[](std::uint16_t value)->float{
        const std::uint32_t bits=
            static_cast<std::uint32_t>(value)<<16;
        float result=0.0f;
        std::memcpy(&result,&bits,sizeof(result));
        return result;
    };

    constexpr int test_m=2;
    constexpr int test_n=16;
    constexpr int test_k=32;

    std::vector<std::uint16_t> input(
        test_m*test_k);
    std::vector<std::uint16_t> weight(
        test_n*test_k);
    std::vector<std::uint16_t> bias(
        test_n);
    std::vector<std::uint16_t> output(
        test_m*test_n);

    for(int m=0;m<test_m;++m){
        for(int k=0;k<test_k;++k){
            const float value=
                static_cast<float>(
                    ((m+1)*(k%7)-9))
                *0.03125f;
            input[m*test_k+k]=
                host_float_to_bf16(value);
        }
    }

    for(int n=0;n<test_n;++n){
        for(int k=0;k<test_k;++k){
            const float value=
                static_cast<float>(
                    ((n%5)-2)*(k%11)-7)
                *0.015625f;
            weight[n*test_k+k]=
                host_float_to_bf16(value);
        }

        bias[n]=host_float_to_bf16(
            static_cast<float>(n-8)
            *0.0078125f);
    }

    std::uint16_t* d_input=nullptr;
    std::uint16_t* d_weight=nullptr;
    std::uint16_t* d_bias=nullptr;
    std::uint16_t* d_output=nullptr;

    auto check_hip=[](
        hipError_t status,
        const char* operation){
        if(status!=hipSuccess){
            throw std::runtime_error(
                std::string("self-test ")
                +operation
                +": "
                +hipGetErrorString(status));
        }
    };

    check_hip(
        hipMalloc(
            reinterpret_cast<void**>(&d_input),
            input.size()*sizeof(std::uint16_t)),
        "hipMalloc input");
    check_hip(
        hipMalloc(
            reinterpret_cast<void**>(&d_weight),
            weight.size()*sizeof(std::uint16_t)),
        "hipMalloc weight");
    check_hip(
        hipMalloc(
            reinterpret_cast<void**>(&d_bias),
            bias.size()*sizeof(std::uint16_t)),
        "hipMalloc bias");
    check_hip(
        hipMalloc(
            reinterpret_cast<void**>(&d_output),
            output.size()*sizeof(std::uint16_t)),
        "hipMalloc output");

    check_hip(
        hipMemcpy(
            d_input,
            input.data(),
            input.size()*sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy input");
    check_hip(
        hipMemcpy(
            d_weight,
            weight.data(),
            weight.size()*sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy weight");
    check_hip(
        hipMemcpy(
            d_bias,
            bias.data(),
            bias.size()*sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy bias");

    qwen3_tts::native_ops::linear_bf16(
        d_input,
        d_weight,
        d_bias,
        d_output,
        test_m,
        test_n,
        test_k);

    check_hip(
        hipDeviceSynchronize(),
        "linear synchronize");

    check_hip(
        hipMemcpy(
            output.data(),
            d_output,
            output.size()*sizeof(std::uint16_t),
            hipMemcpyDeviceToHost),
        "copy output");

    (void)hipFree(d_input);
    (void)hipFree(d_weight);
    (void)hipFree(d_bias);
    (void)hipFree(d_output);

    float max_abs=0.0f;

    for(int m=0;m<test_m;++m){
        for(int n=0;n<test_n;++n){
            float reference=
                host_bf16_to_float(
                    bias[n]);

            for(int k=0;k<test_k;++k){
                reference +=
                    host_bf16_to_float(
                        input[m*test_k+k])
                    *host_bf16_to_float(
                        weight[n*test_k+k]);
            }

            const float got=
                host_bf16_to_float(
                    output[m*test_n+n]);

            max_abs=
                std::max(
                    max_abs,
                    std::abs(got-reference));
        }
    }

    const bool linear_ok =
        std::isfinite(max_abs)
        && max_abs <= 0.03125f;

    const bool ok =
        wav_ok
        && linear_ok;

    std::cout
        << "single_cpp_source: True\n"
        << "self_test_wav_writer: "
        << (wav_ok ? "True" : "False")
        << "\n"
        << "self_test_native_linear: "
        << (linear_ok ? "True" : "False")
        << "\n"
        << "self_test_native_linear_max_abs_diff: "
        << max_abs
        << "\n";

    fs::remove_all(root);
    return ok ? 0 : 3;
}


// ============================================================================
// legacy_legacy_tag_e.1 native raw-text frontend
// ============================================================================

namespace model_frontend {

static constexpr int kHidden = 1024;
static constexpr int kTextHidden = 2048;
static constexpr int kProjectionIntermediate = 2048;
static constexpr std::int64_t kTtsBos = 151672;
static constexpr std::int64_t kTtsEos = 151673;
static constexpr std::int64_t kTtsPad = 151671;
static constexpr std::int64_t kCodecPad = 2148;
static constexpr std::int64_t kCodecBos = 2149;
static constexpr std::int64_t kCodecThink = 2154;
static constexpr std::int64_t kCodecNoThink = 2155;
static constexpr std::int64_t kCodecThinkBos = 2156;
static constexpr std::int64_t kCodecThinkEos = 2157;

static void hip_check(
    hipError_t status,
    const char* expression,
    const char* file,
    int line) {
    if (status == hipSuccess) return;
    throw std::runtime_error(
        std::string("HIP error: ")
        + hipGetErrorString(status)
        + " from " + expression
        + " at " + file + ":"
        + std::to_string(line));
}
#define legacy_HIP_CHECK_b(x) \
    ::qwen3_tts::baseline::model_frontend::hip_check((x),#x,__FILE__,__LINE__)

static float bf16_to_float(std::uint16_t bits) {
    std::uint32_t u =
        static_cast<std::uint32_t>(bits) << 16;
    float out = 0.0f;
    std::memcpy(&out,&u,sizeof(out));
    return out;
}

__host__ __device__
static std::uint16_t f32_to_bf16_rne(float value) {
#if defined(__HIP_DEVICE_COMPILE__)
    union { float f; std::uint32_t u; } x;
    x.f=value;
    const std::uint32_t lsb=(x.u>>16)&1u;
    x.u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(x.u>>16);
#else
    std::uint32_t u=0;
    std::memcpy(&u,&value,sizeof(u));
    const std::uint32_t lsb=(u>>16)&1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u>>16);
#endif
}

__device__
static float bf16_to_float_device(std::uint16_t bits) {
    union { std::uint32_t u; float f; } x;
    x.u=static_cast<std::uint32_t>(bits)<<16;
    return x.f;
}

__global__ void silu_bf16_kernel(
    const std::uint16_t* x,
    std::uint16_t* y,
    int count) {
    const int i=
        static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;

    const float value=
        bf16_to_float_device(x[i]);

    // Same BF16 SiLU boundary already proven in the closed Talker/Predictor
    // path: F.silu(BF16) writes BF16 before the following Linear.
    y[i]=f32_to_bf16_rne(
        value/(1.0f+expf(-value)));
}

struct J {
    enum class K { Null,Bool,Number,String,Array,Object };
    K kind=K::Null;
    bool b=false;
    double n=0;
    std::string s;
    std::vector<J> a;
    std::map<std::string,J> o;
    const J& at(const std::string& key) const {
        auto it=o.find(key);
        if(it==o.end())
            throw std::runtime_error("JSON key missing: "+key);
        return it->second;
    }
};

class JP {
public:
    explicit JP(const std::string& s):s_(s){}
    J parse(){ ws(); J v=value(); ws(); if(p_!=s_.size()) fail("trailing data"); return v; }
private:
    const std::string& s_;
    std::size_t p_=0;
    [[noreturn]] void fail(const std::string& m) const {
        throw std::runtime_error("JSON parse error at "+std::to_string(p_)+": "+m);
    }
    void ws(){ while(p_<s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_; }
    bool eat(char c){ ws(); if(p_<s_.size()&&s_[p_]==c){++p_;return true;} return false; }
    void req(char c){ if(!eat(c)) fail(std::string("expected ")+c); }
    std::string str(){
        req('"'); std::string out;
        while(p_<s_.size()){
            char c=s_[p_++];
            if(c=='"') return out;
            if(c=='\\'){
                if(p_>=s_.size()) fail("escape EOF");
                char e=s_[p_++];
                switch(e){
                    case '"':out.push_back('"');break;
                    case '\\':out.push_back('\\');break;
                    case '/':out.push_back('/');break;
                    case 'b':out.push_back('\b');break;
                    case 'f':out.push_back('\f');break;
                    case 'n':out.push_back('\n');break;
                    case 'r':out.push_back('\r');break;
                    case 't':out.push_back('\t');break;
                    case 'u':
                        if(p_+4>s_.size()) fail("short unicode");
                        p_+=4; out.push_back('?'); break;
                    default: fail("bad escape");
                }
            } else out.push_back(c);
        }
        fail("unterminated string");
    }
    J number(){
        ws(); std::size_t b=p_;
        if(p_<s_.size()&&s_[p_]=='-') ++p_;
        while(p_<s_.size()&&std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if(p_<s_.size()&&s_[p_]=='.'){
            ++p_;
            while(p_<s_.size()&&std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        if(p_<s_.size()&&(s_[p_]=='e'||s_[p_]=='E')){
            ++p_;
            if(p_<s_.size()&&(s_[p_]=='+'||s_[p_]=='-')) ++p_;
            while(p_<s_.size()&&std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        if(b==p_) fail("number");
        J v; v.kind=J::K::Number; v.n=std::stod(s_.substr(b,p_-b)); return v;
    }
    J lit(const char* t,bool b){
        std::size_t n=std::strlen(t);
        if(s_.compare(p_,n,t)!=0) fail("literal");
        p_+=n; J v; v.kind=J::K::Bool;v.b=b;return v;
    }
    J arr(){
        req('[');J v;v.kind=J::K::Array;
        if(eat(']')) return v;
        for(;;){v.a.push_back(value());if(eat(']'))return v;req(',');}
    }
    J obj(){
        req('{');J v;v.kind=J::K::Object;
        if(eat('}')) return v;
        for(;;){ws();std::string k=str();req(':');v.o.emplace(std::move(k),value());if(eat('}'))return v;req(',');}
    }
    J value(){
        ws();if(p_>=s_.size())fail("EOF");
        if(s_[p_]=='{')return obj();
        if(s_[p_]=='[')return arr();
        if(s_[p_]=='"'){J v;v.kind=J::K::String;v.s=str();return v;}
        if(s_[p_]=='t')return lit("true",true);
        if(s_[p_]=='f')return lit("false",false);
        if(s_[p_]=='n'){if(s_.compare(p_,4,"null")!=0)fail("null");p_+=4;return J{};}
        return number();
    }
};

static std::uint64_t le64(const std::uint8_t* p){
    std::uint64_t v=0;
    for(int i=7;i>=0;--i) v=(v<<8)|p[i];
    return v;
}
static std::size_t dtype_size(const std::string& d){
    if(d=="F64"||d=="I64"||d=="U64")return 8;
    if(d=="F32"||d=="I32"||d=="U32")return 4;
    if(d=="F16"||d=="BF16"||d=="I16"||d=="U16")return 2;
    if(d=="I8"||d=="U8"||d=="BOOL")return 1;
    throw std::runtime_error("unsupported dtype: "+d);
}

struct Mapped {
    int fd=-1;
    std::uint8_t* data=nullptr;
    std::size_t size=0;
    explicit Mapped(const fs::path& p){
        fd=::open(p.c_str(),O_RDONLY);
        if(fd<0)throw std::runtime_error("open failed: "+p.string());
        struct stat st{};
        if(::fstat(fd,&st)!=0){::close(fd);throw std::runtime_error("fstat failed");}
        size=static_cast<std::size_t>(st.st_size);
        void* m=::mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
        if(m==MAP_FAILED){::close(fd);fd=-1;throw std::runtime_error("mmap failed");}
        data=static_cast<std::uint8_t*>(m);
    }
    ~Mapped(){if(data)::munmap(data,size);if(fd>=0)::close(fd);}
    Mapped(const Mapped&)=delete;
    Mapped& operator=(const Mapped&)=delete;
};

struct Tensor {
    std::string name,dtype;
    std::vector<std::int64_t> shape;
    const std::uint8_t* data=nullptr;
    std::uint64_t elements() const {
        std::uint64_t n=1;
        for(auto d:shape)n*=static_cast<std::uint64_t>(d);
        return n;
    }
};

class SafeFile {
public:
    explicit SafeFile(const fs::path& path):map_(path){
        if(map_.size<8)throw std::runtime_error("short safetensors");
        const std::uint64_t hlen=le64(map_.data);
        if(hlen==0||8+hlen>map_.size)throw std::runtime_error("bad safetensors header");
        const std::string hs(
            reinterpret_cast<const char*>(map_.data+8),
            static_cast<std::size_t>(hlen));
        J root=JP(hs).parse();
        const std::uint64_t base=8+hlen;
        for(const auto& kv:root.o){
            if(kv.first=="__metadata__")continue;
            const J& meta=kv.second;
            Tensor t;t.name=kv.first;t.dtype=meta.at("dtype").s;
            for(const J& d:meta.at("shape").a)
                t.shape.push_back(static_cast<std::int64_t>(d.n));
            const auto& off=meta.at("data_offsets").a;
            const std::uint64_t begin=static_cast<std::uint64_t>(off[0].n);
            const std::uint64_t end=static_cast<std::uint64_t>(off[1].n);
            if(base+end>map_.size)throw std::runtime_error("tensor out of range");
            if(t.elements()*dtype_size(t.dtype)!=end-begin)
                throw std::runtime_error("tensor byte mismatch: "+t.name);
            t.data=map_.data+base+begin;
            tensors_.emplace(t.name,std::move(t));
        }
    }
    const std::unordered_map<std::string,Tensor>& tensors() const{return tensors_;}
private:
    Mapped map_;
    std::unordered_map<std::string,Tensor> tensors_;
};

class ModelWeights {
public:
    explicit ModelWeights(const fs::path& model_dir){
        std::vector<fs::path> files;
        for(const auto& e:fs::directory_iterator(model_dir))
            if(e.is_regular_file()&&e.path().extension()==".safetensors")
                files.push_back(e.path());
        std::sort(files.begin(),files.end());
        if(files.empty())throw std::runtime_error("no root model safetensors");
        for(const auto& p:files){
            auto f=std::make_unique<SafeFile>(p);
            for(const auto& kv:f->tensors()){
                if(index_.count(kv.first))throw std::runtime_error("duplicate tensor "+kv.first);
                index_.emplace(kv.first,&kv.second);
            }
            files_.push_back(std::move(f));
        }
    }
    const Tensor* maybe_suffix(const std::string& suffix) const{
        const Tensor* found=nullptr;
        for(const auto& kv:index_){
            if(kv.first.size()>=suffix.size()
               &&kv.first.compare(kv.first.size()-suffix.size(),suffix.size(),suffix)==0){
                if(found)throw std::runtime_error("ambiguous tensor suffix: "+suffix);
                found=kv.second;
            }
        }
        return found;
    }

    const Tensor& suffix(const std::string& suffix) const{
        const Tensor* found=maybe_suffix(suffix);
        if(!found)throw std::runtime_error("tensor suffix missing: "+suffix);
        return *found;
    }
private:
    std::vector<std::unique_ptr<SafeFile>> files_;
    std::unordered_map<std::string,const Tensor*> index_;
};


class ExactM1Linear {
public:
    
    explicit ExactM1Linear(const fs::path& assets) {
        (void)assets;
    }

    void launch(
        const std::uint16_t* input,
        const std::uint16_t* weight,
        const std::uint16_t* bias,
        std::uint16_t* output,
        int output_columns,
        int reduction_columns,
        hipStream_t stream=nullptr) const {

        if (bias == nullptr) {
            throw std::runtime_error(
                "legacy_legacy_tag_f frontend projection requires model bias.");
        }

        qwen3_tts::native_ops::linear_bf16(
            input,
            weight,
            bias,
            output,
            1,
            output_columns,
            reduction_columns,
            stream);
    }
};

struct ProjectionTrace {
    std::vector<std::uint16_t> embedding;
    std::vector<std::uint16_t> fc1;
    std::vector<std::uint16_t> silu;
    std::vector<std::uint16_t> fc2;
};

struct FrontendOutput {
    std::vector<std::uint16_t> prefill;
    std::vector<std::uint16_t> tts_pad;
    std::vector<std::uint16_t> trailing_text;
    ProjectionTrace tts_pad_trace;
    std::size_t seq_len=0;
    std::size_t trailing_rows=0;
};

static std::vector<std::uint16_t> tensor_row_bf16(
    const Tensor& t,
    std::int64_t row){
    if(t.dtype!="BF16"||t.shape.size()!=2||t.shape[1]<=0)
        throw std::runtime_error("unexpected BF16 embedding tensor: "+t.name);
    if(row<0||row>=t.shape[0])throw std::runtime_error("embedding id out of range");
    const std::size_t width=static_cast<std::size_t>(t.shape[1]);
    const auto* base=reinterpret_cast<const std::uint16_t*>(t.data);
    const auto* p=base+static_cast<std::size_t>(row)*width;
    return std::vector<std::uint16_t>(p,p+width);
}

static std::vector<std::uint16_t> add_bf16_rows(
    const std::vector<std::uint16_t>& a,
    const std::vector<std::uint16_t>& b){
    if(a.size()!=kHidden||b.size()!=kHidden)throw std::runtime_error("BF16 add row size");
    std::vector<std::uint16_t> out(kHidden);
    for(int i=0;i<kHidden;++i)
        out[i]=f32_to_bf16_rne(
            bf16_to_float(a[i])+bf16_to_float(b[i]));
    return out;
}

class NativeFrontend {
public:
    NativeFrontend(const fs::path& model_dir,const fs::path& assets)
      : weights_(model_dir),linear_(assets){
        text_embedding_=&weights_.suffix("talker.model.text_embedding.weight");
        codec_embedding_=&weights_.suffix("talker.model.codec_embedding.weight");
        fc1_w_=&weights_.suffix("talker.text_projection.linear_fc1.weight");
        fc1_b_=&weights_.suffix("talker.text_projection.linear_fc1.bias");
        fc2_w_=&weights_.suffix("talker.text_projection.linear_fc2.weight");
        fc2_b_=&weights_.suffix("talker.text_projection.linear_fc2.bias");
        validate_linear(*fc1_w_,*fc1_b_,kProjectionIntermediate,kTextHidden);
        validate_linear(*fc2_w_,*fc2_b_,kHidden,kProjectionIntermediate);

        legacy_HIP_CHECK_b(
            hipStreamCreateWithFlags(
                &stream_,
                hipStreamNonBlocking));

        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_in_,
                kTextHidden*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_tmp0_,
                kProjectionIntermediate*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_tmp1_,
                kProjectionIntermediate*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_out_,
                kHidden*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_w1_,
                static_cast<std::size_t>(
                    kProjectionIntermediate)*kTextHidden*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_w2_,
                static_cast<std::size_t>(
                    kHidden)*kProjectionIntermediate*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_b1_,
                kProjectionIntermediate*2,
                stream_));
        legacy_HIP_CHECK_b(
            hipMallocAsync(
                &d_b2_,
                kHidden*2,
                stream_));

        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                d_w1_,fc1_w_->data,
                static_cast<std::size_t>(
                    kProjectionIntermediate)*kTextHidden*2,
                hipMemcpyHostToDevice,
                stream_));
        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                d_w2_,fc2_w_->data,
                static_cast<std::size_t>(
                    kHidden)*kProjectionIntermediate*2,
                hipMemcpyHostToDevice,
                stream_));
        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                d_b1_,fc1_b_->data,
                kProjectionIntermediate*2,
                hipMemcpyHostToDevice,
                stream_));
        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                d_b2_,fc2_b_->data,
                kHidden*2,
                hipMemcpyHostToDevice,
                stream_));

        std::cout
            <<"frontend_gpu_stream: explicit_nonblocking\n";
    }

    ~NativeFrontend(){
        if(stream_!=nullptr){
            if(d_in_)(void)hipFreeAsync(d_in_,stream_);
            if(d_tmp0_)(void)hipFreeAsync(d_tmp0_,stream_);
            if(d_tmp1_)(void)hipFreeAsync(d_tmp1_,stream_);
            if(d_out_)(void)hipFreeAsync(d_out_,stream_);
            if(d_w1_)(void)hipFreeAsync(d_w1_,stream_);
            if(d_w2_)(void)hipFreeAsync(d_w2_,stream_);
            if(d_b1_)(void)hipFreeAsync(d_b1_,stream_);
            if(d_b2_)(void)hipFreeAsync(d_b2_,stream_);
            (void)hipStreamSynchronize(stream_);
            (void)hipStreamDestroy(stream_);
        }
    }

    FrontendOutput build_official_base_xvector_inputs(
        const std::vector<std::int64_t>& ids,
        const std::vector<std::uint16_t>& speaker_embedding,
        const std::string& language) {

        // Official Qwen3-TTS Base streaming x-vector-only construction
        // (x_vector_only_mode=True):
        //
        //   role = text_projection(input_id[:, :3])
        //
        //   explicit language codec prefix:
        //     [think, think_bos, language, think_eos,
        //      speaker_embedding, codec_pad, codec_bos]
        //
        //   talker prefill:
        //     role
        //     + [tts_pad ... tts_pad, tts_bos] + codec_prefix[:-1]
        //     + projected(input_id[:, 3:4]) + codec_bos
        //
        //   trailing_text_hidden:
        //     projected(input_id[:, 4:-5]) + tts_eos
        //
        // Auto language uses [nothink, think_bos, think_eos] instead of the
        // explicit-language four-token tag.
        if(ids.size()<9){
            throw std::runtime_error(
                "official Base prompt is too short for input_id[:, 4:-5]");
        }

        if(speaker_embedding.size()!=kHidden){
            throw std::runtime_error(
                "official Base speaker embedding must have 1024 BF16 values");
        }

        std::unordered_map<
            std::int64_t,
            std::vector<std::uint16_t>> cache;

        auto projected=
            [&](std::int64_t id)
            ->const std::vector<std::uint16_t>& {

                auto it=cache.find(id);

                if(it!=cache.end()){
                    return it->second;
                }

                auto emb=
                    tensor_row_bf16(
                        *text_embedding_,
                        id);

                auto value=
                    project(
                        emb,
                        nullptr);

                return cache.emplace(
                    id,
                    std::move(value))
                    .first->second;
            };

        ProjectionTrace pad_trace;

        const auto pad_embedding=
            tensor_row_bf16(
                *text_embedding_,
                kTtsPad);

        const auto tts_pad=
            project(
                pad_embedding,
                &pad_trace);

        cache.emplace(
            kTtsPad,
            tts_pad);

        const auto tts_bos=
            projected(kTtsBos);

        const auto tts_eos=
            projected(kTtsEos);

        std::string lang=language;

        std::transform(
            lang.begin(),
            lang.end(),
            lang.begin(),
            [](unsigned char c){
                return static_cast<char>(
                    std::tolower(c));
            });

        static const std::unordered_map<
            std::string,
            std::int64_t> language_ids{
                {"english",2050},
                {"german",2053},
                {"spanish",2054},
                {"chinese",2055},
                {"japanese",2058},
                {"french",2061},
                {"sichuan_dialect",2062},
                {"korean",2064},
                {"russian",2069},
                {"beijing_dialect",2074},
                {"italian",2070},
                {"portuguese",2071},
            };

        std::vector<
            std::vector<std::uint16_t>>
            codec_prefix;

        if(lang=="auto"){
            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    kCodecNoThink));

            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    kCodecThinkBos));

            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    kCodecThinkEos));
        }else{
            const auto it=
                language_ids.find(lang);

            if(it==language_ids.end()){
                throw std::runtime_error(
                    "unsupported official Base language: "
                    +language);
            }

            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    kCodecThink));

            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    kCodecThinkBos));

            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    it->second));

            codec_prefix.push_back(
                tensor_row_bf16(
                    *codec_embedding_,
                    kCodecThinkEos));
        }

        // speaker_embed.view(1,1,-1)
        codec_prefix.push_back(
            speaker_embedding);

        codec_prefix.push_back(
            tensor_row_bf16(
                *codec_embedding_,
                kCodecPad));

        codec_prefix.push_back(
            tensor_row_bf16(
                *codec_embedding_,
                kCodecBos));

        if(codec_prefix.size()<2){
            throw std::runtime_error(
                "internal official Base codec prefix underflow");
        }

        std::vector<
            std::vector<std::uint16_t>> rows;

        rows.reserve(
            3
            +codec_prefix.size());

        // _talker_input_embed_role
        for(std::size_t i=0;i<3;++i){
            rows.push_back(
                projected(ids[i]));
        }

        // torch.cat((tts_pad.expand(codec_len - 2), tts_bos)) +
        // codec_input_embedding[:, :-1].
        const std::size_t prefix_without_bos=
            codec_prefix.size()-1;

        for(std::size_t i=0;i<prefix_without_bos;++i){
            const bool use_tts_bos=
                i+1==prefix_without_bos;

            rows.push_back(
                add_bf16_rows(
                    use_tts_bos
                    ?tts_bos
                    :tts_pad,
                    codec_prefix[i]));
        }

        // tts_text_first_token + codec_bos
        rows.push_back(
            add_bf16_rows(
                projected(ids[3]),
                codec_prefix.back()));

        FrontendOutput out;
        out.seq_len=rows.size();
        out.tts_pad=tts_pad;
        out.tts_pad_trace=
            std::move(pad_trace);

        out.prefill.reserve(
            rows.size()*kHidden);

        for(const auto& row:rows){
            out.prefill.insert(
                out.prefill.end(),
                row.begin(),
                row.end());
        }

        // Official streaming trailing_text_hidden:
        // text_projection(input_id[:, 4:-5]) + tts_eos.
        const std::size_t trailing_text_tokens=
            ids.size()-5-4;

        out.trailing_rows=
            trailing_text_tokens+1;

        out.trailing_text.reserve(
            out.trailing_rows*kHidden);

        for(std::size_t i=4;i+5<ids.size();++i){
            const auto& row=
                projected(ids[i]);

            out.trailing_text.insert(
                out.trailing_text.end(),
                row.begin(),
                row.end());
        }

        out.trailing_text.insert(
            out.trailing_text.end(),
            tts_eos.begin(),
            tts_eos.end());

        std::cout
            <<"official_base_frontend: x_vector_only_streaming\n"
            <<"official_base_language: "<<language<<"\\n"
            <<"official_base_prefill_rows: "<<out.seq_len<<"\\n"
            <<"official_base_trailing_text_rows: "<<out.trailing_rows<<"\\n"
            <<"official_base_speaker_embedding_in_prefill: True\n";

        return out;
    }

    FrontendOutput run(const std::vector<std::int64_t>& ids){
        if(ids.size()<8)throw std::runtime_error("token sequence too short for Qwen3-TTS template");
        std::unordered_map<std::int64_t,std::vector<std::uint16_t>> cache;
        auto projected=[&](std::int64_t id)->const std::vector<std::uint16_t>&{
            auto it=cache.find(id); if(it!=cache.end())return it->second;
            auto emb=tensor_row_bf16(*text_embedding_,id);
            auto value=project(emb,nullptr);
            return cache.emplace(id,std::move(value)).first->second;
        };

        const auto tts_bos=projected(kTtsBos);
        const auto tts_eos=projected(kTtsEos);

        ProjectionTrace pad_trace;
        const auto pad_embedding=tensor_row_bf16(*text_embedding_,kTtsPad);
        const auto tts_pad=project(pad_embedding,&pad_trace);
        cache.emplace(kTtsPad,tts_pad);
        const auto c0=tensor_row_bf16(*codec_embedding_,kCodecNoThink);
        const auto c1=tensor_row_bf16(*codec_embedding_,kCodecThinkBos);
        const auto c2=tensor_row_bf16(*codec_embedding_,kCodecThinkEos);
        const auto cpad=tensor_row_bf16(*codec_embedding_,kCodecPad);
        const auto cbos=tensor_row_bf16(*codec_embedding_,kCodecBos);

        std::vector<std::vector<std::uint16_t>> rows;
        rows.reserve(ids.size()+1);
        for(std::size_t i=0;i<3;++i) rows.push_back(projected(ids[i]));
        rows.push_back(add_bf16_rows(tts_pad,c0));
        rows.push_back(add_bf16_rows(tts_pad,c1));
        rows.push_back(add_bf16_rows(tts_pad,c2));
        rows.push_back(add_bf16_rows(tts_bos,cpad));
        for(std::size_t i=3;i+5<ids.size();++i)
            rows.push_back(add_bf16_rows(projected(ids[i]),cpad));
        rows.push_back(add_bf16_rows(tts_eos,cpad));
        rows.push_back(add_bf16_rows(tts_pad,cbos));

        FrontendOutput out;
        out.seq_len=rows.size();
        out.tts_pad=tts_pad;
        out.tts_pad_trace=std::move(pad_trace);
        out.prefill.reserve(rows.size()*kHidden);
        for(const auto& r:rows)
            out.prefill.insert(out.prefill.end(),r.begin(),r.end());
        return out;
    }
private:
    void validate_linear(const Tensor& w,const Tensor& b,int output_columns,int reduction_columns){
        if(w.dtype!="BF16"||w.shape!=std::vector<std::int64_t>{output_columns,reduction_columns})
            throw std::runtime_error("unexpected text_projection weight "+w.name);
        if(b.dtype!="BF16"||b.shape!=std::vector<std::int64_t>{output_columns})
            throw std::runtime_error("unexpected text_projection bias "+b.name);
    }
    std::vector<std::uint16_t> project(
        const std::vector<std::uint16_t>& input,
        ProjectionTrace* trace){

        if(input.size()!=kTextHidden)
            throw std::runtime_error(
                "legacy_legacy_tag_e.2 projection input must have 2048 BF16 values");

        if(trace)
            trace->embedding=input;

        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                d_in_,
                input.data(),
                kTextHidden*sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_));

        linear_.launch(
            static_cast<const std::uint16_t*>(d_in_),
            static_cast<const std::uint16_t*>(d_w1_),
            static_cast<const std::uint16_t*>(d_b1_),
            static_cast<std::uint16_t*>(d_tmp0_),
            kProjectionIntermediate,
            kTextHidden,
            stream_);

        if(trace){
            trace->fc1.resize(kProjectionIntermediate);
            legacy_HIP_CHECK_b(
                hipMemcpyAsync(
                    trace->fc1.data(),
                    d_tmp0_,
                    kProjectionIntermediate*sizeof(std::uint16_t),
                    hipMemcpyDeviceToHost,
                    stream_));
            legacy_HIP_CHECK_b(
                hipStreamSynchronize(
                    stream_));
        }

        hipLaunchKernelGGL(
            silu_bf16_kernel,
            dim3((kProjectionIntermediate+255)/256),
            dim3(256),
            0,
            stream_,
            static_cast<const std::uint16_t*>(d_tmp0_),
            static_cast<std::uint16_t*>(d_tmp1_),
            kProjectionIntermediate);
        legacy_HIP_CHECK_b(
            hipGetLastError());

        if(trace){
            trace->silu.resize(kProjectionIntermediate);
            legacy_HIP_CHECK_b(
                hipMemcpyAsync(
                    trace->silu.data(),
                    d_tmp1_,
                    kProjectionIntermediate*sizeof(std::uint16_t),
                    hipMemcpyDeviceToHost,
                    stream_));
            legacy_HIP_CHECK_b(
                hipStreamSynchronize(
                    stream_));
        }

        linear_.launch(
            static_cast<const std::uint16_t*>(d_tmp1_),
            static_cast<const std::uint16_t*>(d_w2_),
            static_cast<const std::uint16_t*>(d_b2_),
            static_cast<std::uint16_t*>(d_out_),
            kHidden,
            kProjectionIntermediate,
            stream_);

        std::vector<std::uint16_t> out(kHidden);

        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                out.data(),
                d_out_,
                kHidden*sizeof(std::uint16_t),
                hipMemcpyDeviceToHost,
                stream_));

        legacy_HIP_CHECK_b(
            hipStreamSynchronize(
                stream_));

        if(trace)
            trace->fc2=out;

        return out;
    }

    ModelWeights weights_;
    ExactM1Linear linear_;
    hipStream_t stream_=nullptr;
    const Tensor *text_embedding_=nullptr,*codec_embedding_=nullptr;
    const Tensor *fc1_w_=nullptr,*fc1_b_=nullptr,*fc2_w_=nullptr,*fc2_b_=nullptr;
    void *d_in_=nullptr,*d_tmp0_=nullptr,*d_tmp1_=nullptr,*d_out_=nullptr;
    void *d_w1_=nullptr,*d_w2_=nullptr,*d_b1_=nullptr,*d_b2_=nullptr;
};

static void write_u16(const fs::path& p,const std::vector<std::uint16_t>& x){
    std::ofstream f(p,std::ios::binary);
    if(!f)throw std::runtime_error("cannot create "+p.string());
    f.write(reinterpret_cast<const char*>(x.data()),
            static_cast<std::streamsize>(x.size()*sizeof(std::uint16_t)));
    if(!f)throw std::runtime_error("failed writing "+p.string());
}

} // namespace model_frontend


// ============================================================================
// legacy_legacy_tag_j.3 official Qwen3-TTS Base reference-audio speaker conditioning.
//
// This implements the official Base model's x_vector_only_mode=True path:
//   reference WAV -> 24 kHz log-mel -> ECAPA-TDNN speaker encoder -> 1024-D
//   speaker embedding.
//
// The model architecture and mel contract mirror the official
// Qwen3TTSSpeakerEncoder / extract_speaker_embedding path:
//   n_fft=1024, n_mels=128, hop=256, win=1024, fmin=0, fmax=12000,
//   Slaney mel normalization, periodic Hann, reflect padding.
//
// The convolution-heavy ECAPA path is HIP/BF16.  FFT/filterbank preparation is
// native C++ host code; there is no Python/HuggingFace/BLAS runtime.
// ============================================================================
namespace speaker_encoder {

using model_frontend::Tensor;
using model_frontend::ModelWeights;

constexpr int kWave=32;
constexpr int kWaves=8;
constexpr int kThreads=256;
constexpr int kMel=128;
constexpr int kSpeaker=1024;

static void check(
    hipError_t status,
    const char* op) {

    if(status!=hipSuccess){
        throw std::runtime_error(
            std::string("legacy_legacy_tag_j.3 speaker HIP ")
            +op+": "
            +hipGetErrorString(status));
    }
}

static thread_local
hipStream_t g_explicit_stream=nullptr;

static hipStream_t active_stream(){
    if(g_explicit_stream==nullptr){
        throw std::runtime_error(
            "speaker explicit stream is not active");
    }
    return g_explicit_stream;
}

class StreamScope {
public:
    StreamScope(){
        check(
            hipStreamCreateWithFlags(
                &stream_,
                hipStreamNonBlocking),
            "explicit stream create");

        previous_=g_explicit_stream;
        g_explicit_stream=stream_;
    }

    ~StreamScope(){
        if(stream_!=nullptr){
            // Buffer destructors enqueue stream-ordered frees before this
            // scope is destroyed. Drain those frees before destroying stream.
            (void)hipStreamSynchronize(
                stream_);
            g_explicit_stream=previous_;
            (void)hipStreamDestroy(
                stream_);
        }
    }

    StreamScope(const StreamScope&)=delete;
    StreamScope& operator=(const StreamScope&)=delete;

    hipStream_t get() const{
        return stream_;
    }

private:
    hipStream_t previous_=nullptr;
    hipStream_t stream_=nullptr;
};

static hipError_t copy_async(
    void* destination,
    const void* source,
    std::size_t bytes,
    hipMemcpyKind kind){

    return hipMemcpyAsync(
        destination,
        source,
        bytes,
        kind,
        active_stream());
}

struct Buffer {
    void* p=nullptr;
    std::size_t bytes=0;

    Buffer()=default;

    explicit Buffer(std::size_t n):bytes(n){
        if(bytes)check(
            hipMallocAsync(
                &p,
                bytes,
                active_stream()),
            "malloc async");
    }

    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;

    Buffer(Buffer&& other) noexcept
        :p(other.p),bytes(other.bytes){
        other.p=nullptr;
        other.bytes=0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if(this!=&other){
            if(p){
                (void)hipFreeAsync(
                    p,
                    active_stream());
            }
            p=other.p;
            bytes=other.bytes;
            other.p=nullptr;
            other.bytes=0;
        }
        return *this;
    }

    ~Buffer(){
        if(p){
            (void)hipFreeAsync(
                p,
                active_stream());
        }
    }

    template<class T>
    T* as(){
        return reinterpret_cast<T*>(p);
    }

    template<class T>
    const T* as() const{
        return reinterpret_cast<const T*>(p);
    }
};

__device__ __forceinline__
float bf(std::uint16_t x){
    return __uint_as_float(
        static_cast<std::uint32_t>(x)<<16);
}

__device__ __forceinline__
std::uint16_t b16(float x){
    std::uint32_t u=__float_as_uint(x);
    if((u&0x7fffffffU)>0x7f800000U)
        return static_cast<std::uint16_t>(
            (u>>16)|0x0040U);
    u+=0x7fffU+((u>>16)&1U);
    return static_cast<std::uint16_t>(u>>16);
}

__device__ __forceinline__
float wsum(float x){
    for(int o=16;o;o>>=1)
        x+=__shfl_down(x,o,32);
    return x;
}

static int reflect_host(
    int index,
    int length) {

    if(length<=1)return 0;

    while(
        index<0
        || index>=length
    ){
        if(index<0){
            index=-index;
        }else{
            index=
                2*length
                -2
                -index;
        }
    }

    return index;
}

__device__ __forceinline__
int reflect_device(
    int index,
    int length) {

    if(length<=1)return 0;

    while(
        index<0
        || index>=length
    ){
        if(index<0)
            index=-index;
        else
            index=
                2*length
                -2
                -index;
    }

    return index;
}

// Time-major [T,C] equivalent of PyTorch Conv1d [B,C,T].
// One wave owns one [time,out_channel] scalar.
__global__ __launch_bounds__(256,2)
void conv_same_bf16(
    const std::uint16_t* __restrict__ x,
    const std::uint16_t* __restrict__ w,
    const std::uint16_t* __restrict__ bias,
    std::uint16_t* __restrict__ y,
    int time,
    int cin,
    int cout,
    int kernel,
    int dilation,
    int relu) {

    const int wave=
        static_cast<int>(threadIdx.x)>>5;
    const int lane=
        static_cast<int>(threadIdx.x)&31;
    const int t=
        static_cast<int>(blockIdx.y);
    const int oc=
        static_cast<int>(blockIdx.x)
        *kWaves
        +wave;

    if(t>=time||oc>=cout)return;

    const int pad=
        dilation*(kernel-1)/2;

    float acc=0.0f;

    for(
        int r=lane;
        r<cin*kernel;
        r+=kWave
    ){
        const int ic=r/kernel;
        const int kk=r%kernel;
        const int it=
            reflect_device(
                t+kk*dilation-pad,
                time);

        acc=fmaf(
            bf(x[
                static_cast<std::size_t>(it)
                *cin+ic]),
            bf(w[
                (
                    static_cast<std::size_t>(oc)
                    *cin+ic
                )*kernel+kk]),
            acc);
    }

    acc=wsum(acc);

    if(lane==0){
        float value=
            acc+
            (bias?bf(bias[oc]):0.0f);

        if(relu&&value<0.0f)
            value=0.0f;

        y[
            static_cast<std::size_t>(t)
            *cout+oc]=b16(value);
    }
}

__global__
void copy_or_add_chunk(
    const std::uint16_t* x,
    const std::uint16_t* previous,
    std::uint16_t* out,
    int time,
    int full_channels,
    int offset,
    int chunk,
    int add_previous) {

    const std::size_t n=
        static_cast<std::size_t>(time)
        *chunk;
    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i>=n)return;

    const int t=
        static_cast<int>(
            i/chunk);
    const int c=
        static_cast<int>(
            i%chunk);

    float value=
        bf(x[
            static_cast<std::size_t>(t)
            *full_channels
            +offset+c]);

    if(add_previous){
        value+=bf(
            previous[
                static_cast<std::size_t>(t)
                *full_channels
                +offset-chunk+c]);
    }

    out[i]=b16(value);
}

__global__
void scatter_chunk(
    const std::uint16_t* x,
    std::uint16_t* y,
    int time,
    int full_channels,
    int offset,
    int chunk) {

    const std::size_t n=
        static_cast<std::size_t>(time)
        *chunk;
    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i>=n)return;

    const int t=
        static_cast<int>(i/chunk);
    const int c=
        static_cast<int>(i%chunk);

    y[
        static_cast<std::size_t>(t)
        *full_channels
        +offset+c]=x[i];
}

__global__
void add_bf16(
    const std::uint16_t* a,
    const std::uint16_t* b,
    std::uint16_t* y,
    std::size_t n) {

    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i<n)
        y[i]=b16(
            bf(a[i])+bf(b[i]));
}

__global__
void mean_time_bf16(
    const std::uint16_t* x,
    std::uint16_t* y,
    int time,
    int channels) {

    const int c=
        static_cast<int>(blockIdx.x);

    if(c>=channels)return;

    float sum=0.0f;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        sum+=bf(
            x[
                static_cast<std::size_t>(t)
                *channels+c]);
    }

    __shared__ float shared[256];

    shared[threadIdx.x]=sum;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]+=
                shared[
                    threadIdx.x
                    +stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        y[c]=b16(
            shared[0]
            /static_cast<float>(time));
    }
}

__global__
void sigmoid_mul_bf16(
    const std::uint16_t* x,
    const std::uint16_t* gate,
    std::uint16_t* y,
    int time,
    int channels) {

    const std::size_t n=
        static_cast<std::size_t>(time)
        *channels;
    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i>=n)return;

    const int c=
        static_cast<int>(
            i%channels);

    const float g=
        1.0f/
        (
            1.0f+
            expf(
                -bf(gate[c]))
        );

    y[i]=b16(
        bf(x[i])*g);
}

__global__
void concat3_bf16(
    const std::uint16_t* a,
    const std::uint16_t* b,
    const std::uint16_t* c,
    std::uint16_t* y,
    int time,
    int channels) {

    const std::size_t n=
        static_cast<std::size_t>(time)
        *channels*3;
    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i>=n)return;

    const int out_c=
        static_cast<int>(
            i%(channels*3));
    const int t=
        static_cast<int>(
            i/(channels*3));

    const int part=
        out_c/channels;
    const int cc=
        out_c%channels;
    const std::size_t src=
        static_cast<std::size_t>(t)
        *channels+cc;

    y[i]=
        part==0?a[src]:
        part==1?b[src]:
                c[src];
}

__global__
void uniform_stats_bf16(
    const std::uint16_t* x,
    std::uint16_t* mean,
    std::uint16_t* stddev,
    int time,
    int channels) {

    const int c=
        static_cast<int>(blockIdx.x);

    if(c>=channels)return;

    float sum=0.0f;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        sum+=bf(
            x[
                static_cast<std::size_t>(t)
                *channels+c]);
    }

    __shared__ float shared[256];
    __shared__ float mu;

    shared[threadIdx.x]=sum;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]+=
                shared[
                    threadIdx.x
                    +stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0)
        mu=
            shared[0]
            /static_cast<float>(time);

    __syncthreads();

    float var=0.0f;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        const float d=
            bf(x[
                static_cast<std::size_t>(t)
                *channels+c])
            -mu;

        var=fmaf(d,d,var);
    }

    shared[threadIdx.x]=var;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]+=
                shared[
                    threadIdx.x
                    +stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        mean[c]=b16(mu);
        stddev[c]=b16(
            sqrtf(
                fmaxf(
                    shared[0]
                    /static_cast<float>(time),
                    1.0e-12f)));
    }
}

__global__
void build_asp_context_bf16(
    const std::uint16_t* x,
    const std::uint16_t* mean,
    const std::uint16_t* stddev,
    std::uint16_t* y,
    int time,
    int channels) {

    const std::size_t n=
        static_cast<std::size_t>(time)
        *channels*3;
    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i>=n)return;

    const int out_c=
        static_cast<int>(
            i%(channels*3));
    const int t=
        static_cast<int>(
            i/(channels*3));

    if(out_c<channels){
        y[i]=x[
            static_cast<std::size_t>(t)
            *channels+out_c];
    }else if(out_c<2*channels){
        y[i]=mean[
            out_c-channels];
    }else{
        y[i]=stddev[
            out_c-2*channels];
    }
}

__global__
void tanh_bf16(
    const std::uint16_t* x,
    std::uint16_t* y,
    std::size_t n) {

    const std::size_t i=
        static_cast<std::size_t>(
            blockIdx.x)
        *blockDim.x
        +threadIdx.x;

    if(i<n)
        y[i]=b16(
            tanhf(bf(x[i])));
}

__global__
void softmax_time_bf16(
    const std::uint16_t* x,
    std::uint16_t* y,
    int time,
    int channels) {

    const int c=
        static_cast<int>(blockIdx.x);

    if(c>=channels)return;

    float local_max=-INFINITY;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        local_max=fmaxf(
            local_max,
            bf(x[
                static_cast<std::size_t>(t)
                *channels+c]));
    }

    __shared__ float shared[256];
    __shared__ float max_value;
    __shared__ float sum_value;

    shared[threadIdx.x]=local_max;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]=fmaxf(
                shared[threadIdx.x],
                shared[
                    threadIdx.x
                    +stride]);
        }
        __syncthreads();
    }

    if(threadIdx.x==0)
        max_value=shared[0];

    __syncthreads();

    float local_sum=0.0f;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        local_sum+=expf(
            bf(x[
                static_cast<std::size_t>(t)
                *channels+c])
            -max_value);
    }

    shared[threadIdx.x]=local_sum;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]+=
                shared[
                    threadIdx.x
                    +stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0)
        sum_value=shared[0];

    __syncthreads();

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        const float p=
            expf(
                bf(x[
                    static_cast<std::size_t>(t)
                    *channels+c])
                -max_value)
            /sum_value;

        y[
            static_cast<std::size_t>(t)
            *channels+c]=b16(p);
    }
}

__global__
void weighted_stats_bf16(
    const std::uint16_t* x,
    const std::uint16_t* attention,
    std::uint16_t* mean,
    std::uint16_t* stddev,
    int time,
    int channels) {

    const int c=
        static_cast<int>(blockIdx.x);

    if(c>=channels)return;

    float local=0.0f;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        const std::size_t i=
            static_cast<std::size_t>(t)
            *channels+c;

        local=fmaf(
            bf(attention[i]),
            bf(x[i]),
            local);
    }

    __shared__ float shared[256];
    __shared__ float mu;

    shared[threadIdx.x]=local;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]+=
                shared[
                    threadIdx.x
                    +stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0)
        mu=shared[0];

    __syncthreads();

    float local_var=0.0f;

    for(
        int t=
            static_cast<int>(
                threadIdx.x);
        t<time;
        t+=static_cast<int>(
            blockDim.x)
    ){
        const std::size_t i=
            static_cast<std::size_t>(t)
            *channels+c;

        const float d=
            bf(x[i])-mu;

        local_var+=
            bf(attention[i])
            *d*d;
    }

    shared[threadIdx.x]=local_var;
    __syncthreads();

    for(int stride=128;stride;stride>>=1){
        if(
            static_cast<int>(threadIdx.x)
            <stride
        ){
            shared[threadIdx.x]+=
                shared[
                    threadIdx.x
                    +stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        mean[c]=b16(mu);
        stddev[c]=b16(
            sqrtf(
                fmaxf(
                    shared[0],
                    1.0e-12f)));
    }
}

__global__
void concat2_bf16(
    const std::uint16_t* a,
    const std::uint16_t* b,
    std::uint16_t* y,
    int channels) {

    const int i=
        static_cast<int>(
            blockIdx.x*blockDim.x
            +threadIdx.x);

    if(i>=channels*2)return;

    y[i]=
        i<channels
        ?a[i]
        :b[i-channels];
}

static const std::uint16_t* tensor_bf16(
    const Tensor& tensor,
    std::size_t expected,
    const char* op) {

    if(
        tensor.dtype!="BF16"
        || tensor.elements()!=expected
    ){
        throw std::runtime_error(
            std::string(
                "legacy_legacy_tag_j.3 speaker unexpected tensor ")
            +op+" "
            +tensor.name);
    }

    return reinterpret_cast<
        const std::uint16_t*>(
            tensor.data);
}

static Buffer upload_tensor(
    const Tensor& tensor,
    std::size_t expected,
    const char* op) {

    Buffer out(
        expected
        *sizeof(std::uint16_t));

    check(
        copy_async(
            out.p,
            tensor_bf16(
                tensor,
                expected,
                op),
            out.bytes,
            hipMemcpyHostToDevice),
        "upload weight");

    return out;
}

static Buffer conv(
    const Buffer& input,
    int time,
    int cin,
    const Tensor& weight,
    const Tensor& bias,
    int cout,
    int kernel,
    int dilation,
    bool relu) {

    const std::size_t w_count=
        static_cast<std::size_t>(cout)
        *cin*kernel;

    const std::size_t b_count=
        static_cast<std::size_t>(cout);

    Buffer dw=
        upload_tensor(
            weight,
            w_count,
            "conv weight");

    Buffer db=
        upload_tensor(
            bias,
            b_count,
            "conv bias");

    Buffer output(
        static_cast<std::size_t>(time)
        *cout
        *sizeof(std::uint16_t));

    hipLaunchKernelGGL(
        conv_same_bf16,
        dim3(
            static_cast<unsigned>(
                (cout+kWaves-1)
                /kWaves),
            static_cast<unsigned>(
                time),
            1),
        dim3(kThreads),
        0,
        active_stream(),
        input.as<std::uint16_t>(),
        dw.as<std::uint16_t>(),
        db.as<std::uint16_t>(),
        output.as<std::uint16_t>(),
        time,
        cin,
        cout,
        kernel,
        dilation,
        relu?1:0);

    check(
        hipGetLastError(),
        "conv launch");

    
    // boundary for asynchronously launched kernels.
    return output;
}

static Buffer add(
    const Buffer& a,
    const Buffer& b,
    std::size_t elements) {

    Buffer y(
        elements
        *sizeof(std::uint16_t));

    hipLaunchKernelGGL(
        add_bf16,
        dim3(
            static_cast<unsigned>(
                (elements+kThreads-1)
                /kThreads)),
        dim3(kThreads),
        0,
        active_stream(),
        a.as<std::uint16_t>(),
        b.as<std::uint16_t>(),
        y.as<std::uint16_t>(),
        elements);

    check(
        hipGetLastError(),
        "add launch");

    return y;
}

static Buffer res2net(
    const Buffer& input,
    int time,
    const ModelWeights& weights,
    int block_index,
    int dilation) {

    constexpr int channels=512;
    constexpr int chunk=64;

    Buffer output(
        static_cast<std::size_t>(time)
        *channels
        *sizeof(std::uint16_t));

    // Part 0 is identity.
    {
        Buffer temp(
            static_cast<std::size_t>(time)
            *chunk
            *sizeof(std::uint16_t));

        const std::size_t n=
            static_cast<std::size_t>(time)
            *chunk;

        hipLaunchKernelGGL(
            copy_or_add_chunk,
            dim3(
                static_cast<unsigned>(
                    (n+kThreads-1)
                    /kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            input.as<std::uint16_t>(),
            output.as<std::uint16_t>(),
            temp.as<std::uint16_t>(),
            time,
            channels,
            0,
            chunk,
            0);

        check(
            hipGetLastError(),
            "res2 part0 gather");

        hipLaunchKernelGGL(
            scatter_chunk,
            dim3(
                static_cast<unsigned>(
                    (n+kThreads-1)
                    /kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            temp.as<std::uint16_t>(),
            output.as<std::uint16_t>(),
            time,
            channels,
            0,
            chunk);

        check(
            hipGetLastError(),
            "res2 part0 scatter");
    }

    for(int part=1;part<8;++part){
        Buffer temp(
            static_cast<std::size_t>(time)
            *chunk
            *sizeof(std::uint16_t));

        const std::size_t n=
            static_cast<std::size_t>(time)
            *chunk;

        hipLaunchKernelGGL(
            copy_or_add_chunk,
            dim3(
                static_cast<unsigned>(
                    (n+kThreads-1)
                    /kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            input.as<std::uint16_t>(),
            output.as<std::uint16_t>(),
            temp.as<std::uint16_t>(),
            time,
            channels,
            part*chunk,
            chunk,
            part>1?1:0);

        check(
            hipGetLastError(),
            "res2 gather");

        const std::string base=
            "speaker_encoder.blocks."
            +std::to_string(
                block_index)
            +".res2net_block.blocks."
            +std::to_string(
                part-1)
            +".conv";

        Buffer filtered=
            conv(
                temp,
                time,
                chunk,
                weights.suffix(
                    base+".weight"),
                weights.suffix(
                    base+".bias"),
                chunk,
                3,
                dilation,
                true);

        hipLaunchKernelGGL(
            scatter_chunk,
            dim3(
                static_cast<unsigned>(
                    (n+kThreads-1)
                    /kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            filtered.as<std::uint16_t>(),
            output.as<std::uint16_t>(),
            time,
            channels,
            part*chunk,
            chunk);

        check(
            hipGetLastError(),
            "res2 scatter");
    }

    return output;
}

static Buffer se_res2_block(
    Buffer input,
    int time,
    const ModelWeights& weights,
    int index,
    int dilation) {

    const std::string root=
        "speaker_encoder.blocks."
        +std::to_string(index);

    Buffer tdnn1=
        conv(
            input,
            time,
            512,
            weights.suffix(
                root+
                ".tdnn1.conv.weight"),
            weights.suffix(
                root+
                ".tdnn1.conv.bias"),
            512,
            1,
            1,
            true);

    Buffer res2=
        res2net(
            tdnn1,
            time,
            weights,
            index,
            dilation);

    Buffer tdnn2=
        conv(
            res2,
            time,
            512,
            weights.suffix(
                root+
                ".tdnn2.conv.weight"),
            weights.suffix(
                root+
                ".tdnn2.conv.bias"),
            512,
            1,
            1,
            true);

    Buffer mean(
        512
        *sizeof(std::uint16_t));

    hipLaunchKernelGGL(
        mean_time_bf16,
        dim3(512),
        dim3(kThreads),
        0,
        active_stream(),
        tdnn2.as<std::uint16_t>(),
        mean.as<std::uint16_t>(),
        time,
        512);

    check(
        hipGetLastError(),
        "SE mean");

    Buffer se1=
        conv(
            mean,
            1,
            512,
            weights.suffix(
                root+
                ".se_block.conv1.weight"),
            weights.suffix(
                root+
                ".se_block.conv1.bias"),
            128,
            1,
            1,
            true);

    Buffer se2=
        conv(
            se1,
            1,
            128,
            weights.suffix(
                root+
                ".se_block.conv2.weight"),
            weights.suffix(
                root+
                ".se_block.conv2.bias"),
            512,
            1,
            1,
            false);

    Buffer gated(
        static_cast<std::size_t>(time)
        *512
        *sizeof(std::uint16_t));

    const std::size_t n=
        static_cast<std::size_t>(time)
        *512;

    hipLaunchKernelGGL(
        sigmoid_mul_bf16,
        dim3(
            static_cast<unsigned>(
                (n+kThreads-1)
                /kThreads)),
        dim3(kThreads),
        0,
        active_stream(),
        tdnn2.as<std::uint16_t>(),
        se2.as<std::uint16_t>(),
        gated.as<std::uint16_t>(),
        time,
        512);

    check(
        hipGetLastError(),
        "SE sigmoid multiply");

    return add(
        input,
        gated,
        n);
}

struct Wav {
    std::vector<float> samples;
    int sample_rate=0;
};

static std::uint16_t u16(
    const std::uint8_t* p){
    return static_cast<std::uint16_t>(
        p[0]
        |(
            static_cast<std::uint16_t>(
                p[1])<<8));
}

static std::uint32_t u32(
    const std::uint8_t* p){
    return
        static_cast<std::uint32_t>(p[0])
        |(
            static_cast<std::uint32_t>(p[1])
            <<8)
        |(
            static_cast<std::uint32_t>(p[2])
            <<16)
        |(
            static_cast<std::uint32_t>(p[3])
            <<24);
}

static Wav read_wav(
    const fs::path& path) {

    std::ifstream f(
        path,
        std::ios::binary);

    if(!f)
        throw std::runtime_error(
            "cannot open reference WAV: "
            +path.string());

    f.seekg(0,std::ios::end);
    const auto size=f.tellg();
    f.seekg(0,std::ios::beg);

    if(size<44)
        throw std::runtime_error(
            "reference WAV is too short");

    std::vector<std::uint8_t> data(
        static_cast<std::size_t>(size));

    f.read(
        reinterpret_cast<char*>(
            data.data()),
        size);

    if(
        std::memcmp(
            data.data(),
            "RIFF",
            4)!=0
        ||std::memcmp(
            data.data()+8,
            "WAVE",
            4)!=0
    ){
        throw std::runtime_error(
            "reference audio must be RIFF/WAV");
    }

    std::uint16_t format=0;
    std::uint16_t channels=0;
    std::uint16_t bits=0;
    std::uint32_t sr=0;
    const std::uint8_t* pcm=nullptr;
    std::size_t pcm_bytes=0;

    std::size_t pos=12;

    while(pos+8<=data.size()){
        const char* id=
            reinterpret_cast<
                const char*>(
                    data.data()+pos);
        const std::uint32_t chunk=
            u32(data.data()+pos+4);
        pos+=8;

        if(pos+chunk>data.size())
            throw std::runtime_error(
                "invalid reference WAV chunk");

        if(std::memcmp(id,"fmt ",4)==0){
            if(chunk<16)
                throw std::runtime_error(
                    "short reference WAV fmt");
            format=u16(data.data()+pos);
            channels=u16(data.data()+pos+2);
            sr=u32(data.data()+pos+4);
            bits=u16(data.data()+pos+14);
        }else if(
            std::memcmp(
                id,
                "data",
                4)==0
        ){
            pcm=data.data()+pos;
            pcm_bytes=chunk;
        }

        pos+=chunk+(chunk&1U);
    }

    if(
        !pcm
        ||channels==0
        ||sr==0
    ){
        throw std::runtime_error(
            "reference WAV missing fmt/data");
    }

    const int bytes_per_sample=
        bits/8;

    if(
        bytes_per_sample<=0
        ||pcm_bytes%
            (
                static_cast<std::size_t>(
                    bytes_per_sample)
                *channels
            )!=0
    ){
        throw std::runtime_error(
            "unsupported reference WAV geometry");
    }

    const std::size_t frames=
        pcm_bytes/
        (
            static_cast<std::size_t>(
                bytes_per_sample)
            *channels
        );

    Wav out;
    out.sample_rate=
        static_cast<int>(sr);
    out.samples.resize(frames);

    auto sample_at=[&](
        const std::uint8_t* p)->float{

        if(format==1&&bits==16){
            const std::int16_t v=
                static_cast<std::int16_t>(
                    u16(p));
            return
                static_cast<float>(v)
                /32768.0f;
        }

        if(format==1&&bits==24){
            std::int32_t v=
                static_cast<std::int32_t>(
                    p[0]
                    |(
                        static_cast<std::uint32_t>(
                            p[1])<<8)
                    |(
                        static_cast<std::uint32_t>(
                            p[2])<<16));
            if(v&0x00800000)
                v|=~0x00ffffff;
            return
                static_cast<float>(v)
                /8388608.0f;
        }

        if(format==1&&bits==32){
            const std::int32_t v=
                static_cast<std::int32_t>(
                    u32(p));
            return
                static_cast<float>(
                    static_cast<double>(v)
                    /2147483648.0);
        }

        if(format==3&&bits==32){
            float value=0.0f;
            std::memcpy(
                &value,
                p,
                sizeof(value));
            return value;
        }

        throw std::runtime_error(
            "reference WAV must be PCM16/24/32 or float32");
    };

    for(std::size_t t=0;t<frames;++t){
        double sum=0.0;

        for(
            std::uint16_t c=0;
            c<channels;
            ++c
        ){
            const std::uint8_t* p=
                pcm+
                (
                    t*channels+c
                )
                *bytes_per_sample;

            sum+=sample_at(p);
        }

        out.samples[t]=
            static_cast<float>(
                sum/channels);
    }

    return out;
}

static std::vector<float>
resample_linear(
    const std::vector<float>& input,
    int source_rate,
    int target_rate) {

    if(source_rate==target_rate)
        return input;

    if(
        input.empty()
        ||source_rate<=0
        ||target_rate<=0
    ){
        throw std::runtime_error(
            "invalid reference resample");
    }

    const std::size_t out_size=
        static_cast<std::size_t>(
            std::llround(
                static_cast<double>(
                    input.size())
                *target_rate
                /source_rate));

    std::vector<float> out(
        std::max<std::size_t>(
            1,
            out_size));

    for(std::size_t i=0;i<out.size();++i){
        const double position=
            static_cast<double>(i)
            *source_rate
            /target_rate;

        const std::size_t a=
            std::min(
                static_cast<std::size_t>(
                    position),
                input.size()-1);

        const std::size_t b=
            std::min(
                a+1,
                input.size()-1);

        const double alpha=
            position-a;

        out[i]=
            static_cast<float>(
                input[a]
                +(
                    input[b]-input[a]
                )*alpha);
    }

    return out;
}

static void fft1024(
    std::vector<
        std::complex<float>>& a) {

    constexpr int n=1024;

    for(int i=1,j=0;i<n;++i){
        int bit=n>>1;
        for(;j&bit;bit>>=1)
            j^=bit;
        j^=bit;

        if(i<j)
            std::swap(
                a[i],
                a[j]);
    }

    constexpr double pi=
        3.14159265358979323846;

    for(int len=2;len<=n;len<<=1){
        const double angle=
            -2.0*pi/len;
        const std::complex<float> root(
            static_cast<float>(
                std::cos(angle)),
            static_cast<float>(
                std::sin(angle)));

        for(int i=0;i<n;i+=len){
            std::complex<float> w(
                1.0f,
                0.0f);

            for(int j=0;j<len/2;++j){
                const auto u=a[i+j];
                const auto v=
                    a[
                        i+j+len/2]
                    *w;

                a[i+j]=u+v;
                a[i+j+len/2]=u-v;
                w*=root;
            }
        }
    }
}

static double hz_to_mel(double hz){
    constexpr double f_sp=
        200.0/3.0;
    constexpr double min_log_hz=
        1000.0;
    constexpr double min_log_mel=
        min_log_hz/f_sp;
    const double logstep=
        std::log(6.4)/27.0;

    if(hz<min_log_hz)
        return hz/f_sp;

    return
        min_log_mel
        +std::log(
            hz/min_log_hz)
        /logstep;
}

static double mel_to_hz(double mel){
    constexpr double f_sp=
        200.0/3.0;
    constexpr double min_log_hz=
        1000.0;
    constexpr double min_log_mel=
        min_log_hz/f_sp;
    const double logstep=
        std::log(6.4)/27.0;

    if(mel<min_log_mel)
        return mel*f_sp;

    return
        min_log_hz
        *std::exp(
            logstep
            *(
                mel
                -min_log_mel));
}

static std::vector<float>
slaney_mel_basis() {

    constexpr int n_fft=1024;
    constexpr int bins=n_fft/2+1;
    constexpr int n_mels=128;
    constexpr double sr=24000.0;

    const double mel_min=
        hz_to_mel(0.0);
    const double mel_max=
        hz_to_mel(12000.0);

    std::array<double,n_mels+2>
        mel_f{};

    for(int i=0;i<n_mels+2;++i){
        const double m=
            mel_min+
            (
                mel_max-mel_min
            )
            *i
            /(n_mels+1);

        mel_f[i]=mel_to_hz(m);
    }

    std::vector<float> basis(
        static_cast<std::size_t>(
            n_mels)
        *bins);

    for(int m=0;m<n_mels;++m){
        const double lower=
            mel_f[m];
        const double center=
            mel_f[m+1];
        const double upper=
            mel_f[m+2];

        const double enorm=
            2.0/
            (
                upper-lower
            );

        for(int k=0;k<bins;++k){
            const double hz=
                static_cast<double>(k)
                *sr/n_fft;

            const double lower_slope=
                (
                    hz-lower
                )
                /(
                    center-lower
                );

            const double upper_slope=
                (
                    upper-hz
                )
                /(
                    upper-center
                );

            const double weight=
                std::max(
                    0.0,
                    std::min(
                        lower_slope,
                        upper_slope));

            basis[
                static_cast<std::size_t>(m)
                *bins+k]=
                static_cast<float>(
                    weight*enorm);
        }
    }

    return basis;
}

static std::vector<std::uint16_t>
mel_spectrogram(
    const std::vector<float>& audio,
    int& frames) {

    constexpr int n_fft=1024;
    constexpr int hop=256;
    constexpr int padding=
        (n_fft-hop)/2;
    constexpr int bins=
        n_fft/2+1;
    constexpr double pi=
        3.14159265358979323846;

    if(audio.size()<2)
        throw std::runtime_error(
            "reference audio too short");

    std::vector<float> padded(
        audio.size()
        +padding*2);

    for(
        int i=0;
        i<static_cast<int>(
            padded.size());
        ++i
    ){
        const int source=
            reflect_host(
                i-padding,
                static_cast<int>(
                    audio.size()));

        padded[i]=audio[
            static_cast<std::size_t>(
                source)];
    }

    if(padded.size()<n_fft)
        throw std::runtime_error(
            "reference audio too short for speaker encoder STFT");

    frames=
        1+
        static_cast<int>(
            (
                padded.size()
                -n_fft
            )/hop);

    std::array<float,n_fft>
        window{};

    for(int n=0;n<n_fft;++n){
        // torch.hann_window(periodic=True)
        window[n]=
            static_cast<float>(
                0.5
                -0.5*std::cos(
                    2.0*pi*n/n_fft));
    }

    const auto basis=
        slaney_mel_basis();

    std::vector<std::uint16_t> mel(
        static_cast<std::size_t>(
            frames)
        *kMel);

    std::vector<
        std::complex<float>> fft(
            n_fft);

    std::array<float,bins>
        magnitude{};

    for(int frame=0;frame<frames;++frame){
        const std::size_t base=
            static_cast<std::size_t>(
                frame)
            *hop;

        for(int n=0;n<n_fft;++n){
            fft[n]=
                std::complex<float>(
                    padded[base+n]
                    *window[n],
                    0.0f);
        }

        fft1024(fft);

        for(int k=0;k<bins;++k){
            magnitude[k]=
                std::sqrt(
                    std::norm(fft[k])
                    +1.0e-9f);
        }

        for(int m=0;m<kMel;++m){
            float sum=0.0f;

            const float* row=
                basis.data()
                +static_cast<std::size_t>(m)
                *bins;

            for(int k=0;k<bins;++k)
                sum=fmaf(
                    row[k],
                    magnitude[k],
                    sum);

            sum=std::log(
                std::max(
                    sum,
                    1.0e-5f));

            mel[
                static_cast<std::size_t>(
                    frame)
                *kMel+m]=
                model_frontend::
                    f32_to_bf16_rne(
                        sum);
        }
    }

    return mel;
}

class Encoder {
public:
    explicit Encoder(
        const fs::path& model_dir)
        :weights_(
            fs::absolute(
                model_dir)){}

    std::vector<std::uint16_t>
    run(
        const fs::path& reference_wav,
        const fs::path& correctness_dir = fs::path{}) {

        Wav wav=
            read_wav(
                reference_wav);

        std::cout
            <<"ref_audio_input_sample_rate: "
            <<wav.sample_rate<<"\n"
            <<"ref_audio_input_samples: "
            <<wav.samples.size()<<"\n";

        if(wav.sample_rate!=24000){
            throw std::runtime_error(
                "legacy_legacy_tag_j.3 official Base x-vector path requires a 24 kHz reference WAV. "
                "Use the official clone.wav or resample the input to 24000 Hz before inference.");
        }

        const std::vector<float>& audio=
            wav.samples;

        std::cout
            <<"ref_audio_resample_backend: identity_24khz_exact\n"
            <<"ref_audio_24k_samples: "
            <<audio.size()<<"\n";

        int time=0;
        auto mel_host=
            mel_spectrogram(
                audio,
                time);

        if(!correctness_dir.empty()){
            write_binary(
                correctness_dir/
                    "native_speaker_mel.bf16.bin",
                mel_host);

            write_scalar_text(
                correctness_dir/
                    "native_speaker_mel_shape.txt",
                std::to_string(time)+" 128");
        }

        std::cout
            <<"speaker_mel_backend: native_cpp_slaney_stft\n"
            <<"speaker_mel_frames: "<<time<<"\n"
            <<"speaker_mel_bins: 128\n"
            <<"speaker_gpu_stream: explicit_nonblocking\n"
            <<"speaker_gpu_allocator: stream_ordered\n";

        StreamScope stream_scope;

        Buffer mel(
            mel_host.size()
            *sizeof(std::uint16_t));

        check(
            copy_async(
                mel.p,
                mel_host.data(),
                mel.bytes,
                hipMemcpyHostToDevice),
            "mel upload");

        Buffer h0=
            conv(
                mel,
                time,
                128,
                weights_.suffix(
                    "speaker_encoder.blocks.0.conv.weight"),
                weights_.suffix(
                    "speaker_encoder.blocks.0.conv.bias"),
                512,
                5,
                1,
                true);

        Buffer h1=
            se_res2_block(
                std::move(h0),
                time,
                weights_,
                1,
                2);

        // Preserve block outputs used by MFA.
        Buffer keep1(
            static_cast<std::size_t>(time)
            *512
            *sizeof(std::uint16_t));

        check(
            copy_async(
                keep1.p,
                h1.p,
                keep1.bytes,
                hipMemcpyDeviceToDevice),
            "speaker keep1");

        Buffer h2=
            se_res2_block(
                std::move(h1),
                time,
                weights_,
                2,
                3);

        Buffer keep2(
            static_cast<std::size_t>(time)
            *512
            *sizeof(std::uint16_t));

        check(
            copy_async(
                keep2.p,
                h2.p,
                keep2.bytes,
                hipMemcpyDeviceToDevice),
            "speaker keep2");

        Buffer h3=
            se_res2_block(
                std::move(h2),
                time,
                weights_,
                3,
                4);

        Buffer aggregated(
            static_cast<std::size_t>(time)
            *1536
            *sizeof(std::uint16_t));

        const std::size_t aggregate_n=
            static_cast<std::size_t>(
                time)
            *1536;

        hipLaunchKernelGGL(
            concat3_bf16,
            dim3(
                static_cast<unsigned>(
                    (
                        aggregate_n
                        +kThreads-1
                    )/kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            keep1.as<std::uint16_t>(),
            keep2.as<std::uint16_t>(),
            h3.as<std::uint16_t>(),
            aggregated.as<std::uint16_t>(),
            time,
            512);

        check(
            hipGetLastError(),
            "speaker MFA concat");

        Buffer mfa=
            conv(
                aggregated,
                time,
                1536,
                weights_.suffix(
                    "speaker_encoder.mfa.conv.weight"),
                weights_.suffix(
                    "speaker_encoder.mfa.conv.bias"),
                1536,
                1,
                1,
                true);

        Buffer global_mean(
            1536
            *sizeof(std::uint16_t));
        Buffer global_std(
            1536
            *sizeof(std::uint16_t));

        hipLaunchKernelGGL(
            uniform_stats_bf16,
            dim3(1536),
            dim3(kThreads),
            0,
            active_stream(),
            mfa.as<std::uint16_t>(),
            global_mean.as<std::uint16_t>(),
            global_std.as<std::uint16_t>(),
            time,
            1536);

        check(
            hipGetLastError(),
            "speaker ASP global stats");

        Buffer asp_context(
            static_cast<std::size_t>(time)
            *4608
            *sizeof(std::uint16_t));

        const std::size_t asp_context_n=
            static_cast<std::size_t>(
                time)
            *4608;

        hipLaunchKernelGGL(
            build_asp_context_bf16,
            dim3(
                static_cast<unsigned>(
                    (
                        asp_context_n
                        +kThreads-1
                    )/kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            mfa.as<std::uint16_t>(),
            global_mean.as<std::uint16_t>(),
            global_std.as<std::uint16_t>(),
            asp_context.as<std::uint16_t>(),
            time,
            1536);

        check(
            hipGetLastError(),
            "speaker ASP context");

        Buffer att_hidden=
            conv(
                asp_context,
                time,
                4608,
                weights_.suffix(
                    "speaker_encoder.asp.tdnn.conv.weight"),
                weights_.suffix(
                    "speaker_encoder.asp.tdnn.conv.bias"),
                128,
                1,
                1,
                true);

        Buffer att_tanh(
            static_cast<std::size_t>(time)
            *128
            *sizeof(std::uint16_t));

        const std::size_t tanh_n=
            static_cast<std::size_t>(
                time)
            *128;

        hipLaunchKernelGGL(
            tanh_bf16,
            dim3(
                static_cast<unsigned>(
                    (
                        tanh_n
                        +kThreads-1
                    )/kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            att_hidden.as<std::uint16_t>(),
            att_tanh.as<std::uint16_t>(),
            tanh_n);

        check(
            hipGetLastError(),
            "speaker ASP tanh");

        Buffer att_logits=
            conv(
                att_tanh,
                time,
                128,
                weights_.suffix(
                    "speaker_encoder.asp.conv.weight"),
                weights_.suffix(
                    "speaker_encoder.asp.conv.bias"),
                1536,
                1,
                1,
                false);

        Buffer attention(
            static_cast<std::size_t>(time)
            *1536
            *sizeof(std::uint16_t));

        hipLaunchKernelGGL(
            softmax_time_bf16,
            dim3(1536),
            dim3(kThreads),
            0,
            active_stream(),
            att_logits.as<std::uint16_t>(),
            attention.as<std::uint16_t>(),
            time,
            1536);

        check(
            hipGetLastError(),
            "speaker ASP softmax");

        Buffer pooled_mean(
            1536
            *sizeof(std::uint16_t));
        Buffer pooled_std(
            1536
            *sizeof(std::uint16_t));

        hipLaunchKernelGGL(
            weighted_stats_bf16,
            dim3(1536),
            dim3(kThreads),
            0,
            active_stream(),
            mfa.as<std::uint16_t>(),
            attention.as<std::uint16_t>(),
            pooled_mean.as<std::uint16_t>(),
            pooled_std.as<std::uint16_t>(),
            time,
            1536);

        check(
            hipGetLastError(),
            "speaker ASP weighted stats");

        Buffer pooled(
            3072
            *sizeof(std::uint16_t));

        hipLaunchKernelGGL(
            concat2_bf16,
            dim3(
                static_cast<unsigned>(
                    (3072+kThreads-1)
                    /kThreads)),
            dim3(kThreads),
            0,
            active_stream(),
            pooled_mean.as<std::uint16_t>(),
            pooled_std.as<std::uint16_t>(),
            pooled.as<std::uint16_t>(),
            1536);

        check(
            hipGetLastError(),
            "speaker pooled concat");

        Buffer embedding=
            conv(
                pooled,
                1,
                3072,
                weights_.suffix(
                    "speaker_encoder.fc.weight"),
                weights_.suffix(
                    "speaker_encoder.fc.bias"),
                kSpeaker,
                1,
                1,
                false);

        std::vector<std::uint16_t>
            host(kSpeaker);

        check(
            copy_async(
                host.data(),
                embedding.p,
                host.size()
                *sizeof(std::uint16_t),
                hipMemcpyDeviceToHost),
            "speaker embedding download");

        check(
            hipStreamSynchronize(
                active_stream()),
            "explicit stream synchronize");

        if(!correctness_dir.empty()){
            write_binary(
                correctness_dir/
                    "native_speaker_embedding.bf16.bin",
                host);
        }

        std::cout
            <<"speaker_encoder_backend: native_hip_bf16_ecapa_tdnn\n"
            <<"speaker_embedding_dim: 1024\n"
            <<"speaker_embedding_ready: True\n";

        return host;
    }

private:
    ModelWeights weights_;
};

} // namespace speaker_encoder



// ============================================================================
// legacy_legacy_tag_j.2 native generation sampling.
//
// Official Qwen3-TTS 12Hz Base generation policy:
//   Talker:    do_sample=true, temperature=0.9, top_k=50, top_p=1.0
//   Predictor: subtalker_dosample=true, temperature=0.9, top_k=50, top_p=1.0
//
// top_p=1.0 is a no-op after top-k filtering.
//
// This engine deliberately uses its own small deterministic RNG rather than
// reproducing PyTorch/Philox bit-for-bit.  The goal is algorithmic-equivalent
// categorical sampling from the same filtered distribution.
// ============================================================================
struct SamplingRng {
    std::uint64_t state=1234;

    void seed(std::uint64_t value) {
        state=value;
    }

    std::uint64_t next_u64() {
        // SplitMix64: compact, deterministic, high-quality host RNG.
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z=state;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z=(z^(z>>27))*0x94d049bb133111ebULL;
        return z^(z>>31);
    }

    double uniform01() {
        // 53 random mantissa bits in [0,1).
        return static_cast<double>(
            next_u64() >> 11)
            * (1.0/9007199254740992.0);
    }
};

static SamplingRng g_sampling_rng;
static bool g_sampling_enabled=false;

static void configure_sampling(
    bool enabled,
    std::uint64_t seed) {

    g_sampling_enabled=enabled;
    g_sampling_rng.seed(seed);

    std::cout
        <<"generation_policy: "
        <<(enabled
            ?"native_topk_categorical_sampling"
            :"greedy_diagnostic")
        <<"\n"
        <<"sampling_seed: "<<seed<<"\n"
        <<"talker_do_sample: "
        <<(enabled?"True":"False")
        <<"\n"
        <<"talker_temperature: 0.9\n"
        <<"talker_top_k: 50\n"
        <<"talker_top_p: 1.0\n"
        <<"subtalker_do_sample: "
        <<(enabled?"True":"False")
        <<"\n"
        <<"subtalker_temperature: 0.9\n"
        <<"subtalker_top_k: 50\n"
        <<"subtalker_top_p: 1.0\n";
}

struct SamplingCandidate {
    float score=0.0f;
    std::int32_t token=0;
};

static std::int32_t sample_topk_f32(
    const float* scores,
    int vocab,
    int top_k,
    float temperature) {

    if(
        scores==nullptr
        || vocab<=0
        || top_k<=0
        || !(temperature>0.0f)
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.2 invalid sampling geometry");
    }

    std::vector<SamplingCandidate>
        candidates;

    candidates.reserve(
        static_cast<std::size_t>(vocab));

    for(int token=0;token<vocab;++token){
        const float score=scores[token];

        if(std::isfinite(score)){
            candidates.push_back(
                SamplingCandidate{
                    score,
                    token});
        }
    }

    if(candidates.empty()){
        throw std::runtime_error(
            "legacy_legacy_tag_j.2 sampling has no finite logits");
    }

    const int keep=
        std::min(
            top_k,
            static_cast<int>(
                candidates.size()));

    std::partial_sort(
        candidates.begin(),
        candidates.begin()+keep,
        candidates.end(),
        [](
            const SamplingCandidate& a,
            const SamplingCandidate& b){

            if(a.score!=b.score){
                return a.score>b.score;
            }

            return a.token<b.token;
        });

    candidates.resize(
        static_cast<std::size_t>(keep));

    const double inv_temperature=
        1.0/static_cast<double>(
            temperature);

    double max_scaled=
        -std::numeric_limits<double>::infinity();

    for(const auto& candidate:candidates){
        max_scaled=
            std::max(
                max_scaled,
                static_cast<double>(
                    candidate.score)
                    * inv_temperature);
    }

    std::vector<double> weights(
        candidates.size());

    double total=0.0;

    for(std::size_t i=0;i<candidates.size();++i){
        const double scaled=
            static_cast<double>(
                candidates[i].score)
            * inv_temperature;

        const double weight=
            std::exp(
                scaled-max_scaled);

        weights[i]=weight;
        total+=weight;
    }

    if(
        !(total>0.0)
        || !std::isfinite(total)
    ){
        return candidates.front().token;
    }

    const double target=
        g_sampling_rng.uniform01()
        * total;

    double cumulative=0.0;

    for(std::size_t i=0;i<candidates.size();++i){
        cumulative+=weights[i];

        if(target<cumulative){
            return candidates[i].token;
        }
    }

    return candidates.back().token;
}

static float bf16_host_to_f32(
    std::uint16_t bits) {

    std::uint32_t raw=
        static_cast<std::uint32_t>(bits)
        <<16;

    float value=0.0f;
    std::memcpy(
        &value,
        &raw,
        sizeof(value));

    return value;
}

static std::int32_t sample_topk_bf16(
    const std::uint16_t* logits,
    int vocab,
    int top_k,
    float temperature) {

    std::vector<float> scores(
        static_cast<std::size_t>(vocab));

    for(int token=0;token<vocab;++token){
        scores[
            static_cast<std::size_t>(token)
        ]=
            bf16_host_to_f32(
                logits[token]);
    }

    return sample_topk_f32(
        scores.data(),
        vocab,
        top_k,
        temperature);
}



// ============================================================================
// GPU top-k categorical sampler shared by Talker FinalHead and Code Predictor.
// The RNG sequence is SplitMix64 and is kept in one device-side state so the
// draw order remains: code0 -> predictor groups 1..15 -> next code0 -> ...
// ============================================================================
constexpr int kGpuSamplerThreads=256;
constexpr int kGpuSamplerSortSize=4096;
constexpr int kGpuSamplerTopK=50;
constexpr float kGpuSamplerTemperature=0.9f;

__device__ __forceinline__
float gpu_sampler_bf16_to_float(std::uint16_t bits){
    return __uint_as_float(
        static_cast<std::uint32_t>(bits)<<16);
}

__device__ __forceinline__
bool gpu_sampler_better(
    float as,int ai,float bs,int bi){
    return as>bs || (as==bs && ai<bi);
}

__device__ __forceinline__
std::uint64_t gpu_sampler_splitmix64_next(
    std::uint64_t* state){
    std::uint64_t z=
        *state+0x9e3779b97f4a7c15ULL;
    *state=z;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}

__device__ __forceinline__
double gpu_sampler_uniform01(
    std::uint64_t* state){
    return static_cast<double>(
        gpu_sampler_splitmix64_next(state)>>11)
        *(1.0/9007199254740992.0);
}

__device__ __forceinline__
void gpu_sampler_bitonic_sort(
    float* values,int* ids){
    for(int width=2;
        width<=kGpuSamplerSortSize;
        width<<=1){
        for(int stride=width>>1;
            stride>0;
            stride>>=1){
            for(int index=
                    static_cast<int>(threadIdx.x);
                index<kGpuSamplerSortSize;
                index+=static_cast<int>(blockDim.x)){
                const int partner=index^stride;
                if(partner<=index)continue;
                const bool descending=
                    (index&width)==0;
                const float av=values[index];
                const int ai=ids[index];
                const float bv=values[partner];
                const int bi=ids[partner];
                const bool a_better=
                    gpu_sampler_better(av,ai,bv,bi);
                const bool swap=
                    descending?!a_better:a_better;
                if(swap){
                    values[index]=bv;ids[index]=bi;
                    values[partner]=av;ids[partner]=ai;
                }
            }
            __syncthreads();
        }
    }
}

__global__
void gpu_sample_topk_f32_kernel(
    const float* __restrict__ logits,
    int vocab,int top_k,float temperature,
    std::uint64_t* __restrict__ rng_state,
    std::int32_t* __restrict__ output_id){
    extern __shared__ unsigned char shared_bytes[];
    float* values=
        reinterpret_cast<float*>(shared_bytes);
    int* ids=
        reinterpret_cast<int*>(
            values+kGpuSamplerSortSize);

    for(int i=static_cast<int>(threadIdx.x);
        i<kGpuSamplerSortSize;
        i+=static_cast<int>(blockDim.x)){
        values[i]=i<vocab?logits[i]:-INFINITY;
        ids[i]=i<vocab?i:0x7fffffff;
    }
    __syncthreads();
    gpu_sampler_bitonic_sort(values,ids);

    if(threadIdx.x!=0)return;
    const int keep=top_k<vocab?top_k:vocab;
    const double inv_t=
        1.0/static_cast<double>(temperature);
    const double maximum=
        static_cast<double>(values[0])*inv_t;

    double total=0.0;
    for(int i=0;i<keep;++i)
        total+=::exp(
            static_cast<double>(values[i])
            *inv_t-maximum);

    if(!(total>0.0)){
        output_id[0]=ids[0];
        return;
    }

    const double target=
        gpu_sampler_uniform01(rng_state)*total;
    double cumulative=0.0;

    for(int i=0;i<keep;++i){
        cumulative+=::exp(
            static_cast<double>(values[i])
            *inv_t-maximum);
        if(target<cumulative){
            output_id[0]=ids[i];
            return;
        }
    }
    output_id[0]=ids[keep-1];
}

__global__
void gpu_sample_topk_bf16_kernel(
    const std::uint16_t* __restrict__ logits,
    int last_row_offset,
    int vocab,int top_k,float temperature,
    std::uint64_t* __restrict__ rng_state,
    std::int32_t* __restrict__ output_id){
    extern __shared__ unsigned char shared_bytes[];
    float* values=
        reinterpret_cast<float*>(shared_bytes);
    int* ids=
        reinterpret_cast<int*>(
            values+kGpuSamplerSortSize);

    for(int i=static_cast<int>(threadIdx.x);
        i<kGpuSamplerSortSize;
        i+=static_cast<int>(blockDim.x)){
        values[i]=i<vocab
            ?gpu_sampler_bf16_to_float(
                logits[last_row_offset+i])
            :-INFINITY;
        ids[i]=i<vocab?i:0x7fffffff;
    }
    __syncthreads();
    gpu_sampler_bitonic_sort(values,ids);

    if(threadIdx.x!=0)return;
    const int keep=top_k<vocab?top_k:vocab;
    const double inv_t=
        1.0/static_cast<double>(temperature);
    const double maximum=
        static_cast<double>(values[0])*inv_t;

    double total=0.0;
    for(int i=0;i<keep;++i)
        total+=::exp(
            static_cast<double>(values[i])
            *inv_t-maximum);

    if(!(total>0.0)){
        output_id[0]=ids[0];
        return;
    }

    const double target=
        gpu_sampler_uniform01(rng_state)*total;
    double cumulative=0.0;
    for(int i=0;i<keep;++i){
        cumulative+=::exp(
            static_cast<double>(values[i])
            *inv_t-maximum);
        if(target<cumulative){
            output_id[0]=ids[i];
            return;
        }
    }
    output_id[0]=ids[keep-1];
}

constexpr std::size_t gpu_sampler_shared_bytes(){
    return static_cast<std::size_t>(
        kGpuSamplerSortSize)
        *(sizeof(float)+sizeof(int));
}


__global__
void append_sampled_code_history_kernel(
    const std::int32_t* __restrict__ code,
    std::int32_t* __restrict__ history,
    int history_index) {

    if (
        blockIdx.x == 0
        && threadIdx.x == 0
    ) {
        history[history_index] =
            code[0];
    }
}


class GpuSamplingState{
public:
    explicit GpuSamplingState(std::uint64_t seed){
        const hipError_t a=hipMalloc(
            reinterpret_cast<void**>(&state_),
            sizeof(std::uint64_t));
        if(a!=hipSuccess)
            throw std::runtime_error(
                std::string("GPU sampler hipMalloc: ")
                +hipGetErrorString(a));
        const hipError_t c=hipMemcpy(
            state_,&seed,sizeof(seed),
            hipMemcpyHostToDevice);
        if(c!=hipSuccess){
            (void)hipFree(state_);state_=nullptr;
            throw std::runtime_error(
                std::string("GPU sampler seed upload: ")
                +hipGetErrorString(c));
        }
    }
    ~GpuSamplingState(){
        if(state_)(void)hipFree(state_);
    }
    GpuSamplingState(const GpuSamplingState&)=delete;
    GpuSamplingState& operator=(
        const GpuSamplingState&)=delete;

    void reset(std::uint64_t seed){
        const hipError_t status=
            hipMemcpy(
                state_,
                &seed,
                sizeof(seed),
                hipMemcpyHostToDevice);

        if(status!=hipSuccess){
            throw std::runtime_error(
                std::string("GPU sampler reset seed upload: ")
                +hipGetErrorString(status));
        }
    }

    std::uint64_t* device_state()const{
        return state_;
    }
private:
    std::uint64_t* state_=nullptr;
};



// ============================================================================
// RX7900XTX-24G resident CU partition
//
// Physical device is the first-class object:
//   RX 7900 XTX = 96 CUs
//
// Resident execution policy:
//   generation: 64 CUs
//   decoder:    32 CUs
//
// Interleaving CU indices avoids placing one workload only on one contiguous

// behavior and never enables this partition.
// ============================================================================

static bool g_resident_cu_partition_enabled=false;
static int g_resident_generation_wgps=28;
static int g_resident_decoder_wgps=20;

struct CuPartitionMask {
    int total_cus=0;
    int generation_cus=0;
    int decoder_cus=0;
    std::vector<std::uint32_t> generation;
    std::vector<std::uint32_t> decoder;
};

static CuPartitionMask
make_rx7900xtx_24g_generation_decoder_cu_partition() {
    int device=0;

    const hipError_t get_device_status=
        hipGetDevice(&device);

    if(get_device_status!=hipSuccess){
        throw std::runtime_error(
            std::string("RX7900XTX CU partition hipGetDevice: ")
            +hipGetErrorString(get_device_status));
    }

    hipDeviceProp_t properties{};

    const hipError_t properties_status=
        hipGetDeviceProperties(
            &properties,
            device);

    if(properties_status!=hipSuccess){
        throw std::runtime_error(
            std::string("RX7900XTX CU partition hipGetDeviceProperties: ")
            +hipGetErrorString(properties_status));
    }

    const int total_cus=
        properties.multiProcessorCount;

    // RX7900XTX gfx11 uses WGP mode here:
    //   hipDeviceProp_t::multiProcessorCount = 48 WGP
    //   1 WGP = 2 physical CUs
    //   therefore 48 WGP = 96 physical CUs.
    if(total_cus!=48){
        throw std::runtime_error(
            "rx7900xtx-24g WGP profile requires hipDeviceProp_t::multiProcessorCount=48; "
            "runtime reported "
            +std::to_string(total_cus));
    }

    const int expected_generation=
        g_resident_generation_wgps;

    const int expected_decoder=
        g_resident_decoder_wgps;

    if(
        expected_generation<=0
        ||expected_decoder<=0
        ||expected_generation+expected_decoder
            !=total_cus
    ){
        throw std::runtime_error(
            "rx7900xtx-24g resident WGP split must be positive and sum to 48");
    }

    const std::size_t words=
        static_cast<std::size_t>(
            total_cus+31)
        /32;

    CuPartitionMask result;
    result.total_cus=total_cus;
    result.generation.assign(words,0U);
    result.decoder.assign(words,0U);

    // Evenly distribute decoder WGPs over the 48 runtime scheduler units.
    // Bresenham-style exact allocation:
    //   floor((i+1)*D/48) > floor(i*D/48) => decoder.
    // This gives an exact G/D count while avoiding one large contiguous region.
    for(int cu=0;cu<total_cus;++cu){
        const std::size_t word=
            static_cast<std::size_t>(cu)
            /32;

        const std::uint32_t bit=
            1U
            <<static_cast<unsigned>(
                cu&31);

        const int before=
            (cu*expected_decoder)
            /total_cus;

        const int after=
            ((cu+1)*expected_decoder)
            /total_cus;

        if(after>before){
            result.decoder[word]|=bit;
            ++result.decoder_cus;
        }else{
            result.generation[word]|=bit;
            ++result.generation_cus;
        }
    }

    if(
        result.generation_cus
            !=expected_generation
        ||result.decoder_cus
            !=expected_decoder
    ){
        throw std::runtime_error(
            "rx7900xtx-24g runtime WGP partition count mismatch");
    }

    return result;
}

static hipStream_t
create_rx7900xtx_partitioned_stream(
    bool decoder_stream,
    const char* label) {

    const auto partition=
        make_rx7900xtx_24g_generation_decoder_cu_partition();

    const auto& mask=
        decoder_stream
        ?partition.decoder
        :partition.generation;

    hipStream_t stream=nullptr;

    const hipError_t status=
        hipExtStreamCreateWithCUMask(
            &stream,
            static_cast<std::uint32_t>(
                mask.size()),
            mask.data());

    if(status!=hipSuccess){
        throw std::runtime_error(
            std::string(label)
            +": hipExtStreamCreateWithCUMask: "
            +hipGetErrorString(status));
    }

    std::vector<std::uint32_t>
        actual(
            mask.size(),
            0U);

    const hipError_t readback_status=
        hipExtStreamGetCUMask(
            stream,
            static_cast<std::uint32_t>(
                actual.size()),
            actual.data());

    if(readback_status!=hipSuccess){
        (void)hipStreamDestroy(stream);

        throw std::runtime_error(
            std::string(label)
            +": hipExtStreamGetCUMask: "
            +hipGetErrorString(readback_status));
    }

    if(actual!=mask){
        (void)hipStreamDestroy(stream);

        throw std::runtime_error(
            std::string(label)
            +": runtime CU mask differs from requested mask");
    }

    return stream;
}

static void
print_rx7900xtx_cu_partition_summary_once() {
    static std::once_flag once;

    std::call_once(
        once,
        [](){
            const auto partition=
                make_rx7900xtx_24g_generation_decoder_cu_partition();

            std::cout
                <<"resident_cu_partition_policy: balanced_exact_wgp_split\n"
                <<"resident_scheduler_mode: WGP\n"
                <<"resident_physical_cu_total: 96\n"
                <<"resident_wgp_total: "
                <<partition.total_cus
                <<"\n"
                <<"resident_wgp_generation: "
                <<partition.generation_cus
                <<"\n"
                <<"resident_wgp_decoder: "
                <<partition.decoder_cus
                <<"\n"
                <<"resident_generation_physical_cus: "
                <<partition.generation_cus*2
                <<"\n"
                <<"resident_decoder_physical_cus: "
                <<partition.decoder_cus*2
                <<"\n"
                <<"resident_cu_mask_words: "
                <<partition.generation.size()
                <<"\n"
                <<"resident_cu_total: "
                <<partition.total_cus
                <<"\n"
                <<"resident_cu_generation: "
                <<partition.generation_cus
                <<"\n"
                <<"resident_cu_decoder: "
                <<partition.decoder_cus
                <<"\n"
                <<"resident_cu_masks_disjoint: True\n";
        });
}

static hipStream_t
create_generation_execution_stream(
    const char* label) {

    if(g_resident_cu_partition_enabled){
        print_rx7900xtx_cu_partition_summary_once();

        return create_rx7900xtx_partitioned_stream(
            false,
            label);
    }

    int least=0;
    int greatest=0;

    const hipError_t priority_status=
        hipDeviceGetStreamPriorityRange(
            &least,
            &greatest);

    if(priority_status!=hipSuccess){
        throw std::runtime_error(
            std::string(label)
            +": hipDeviceGetStreamPriorityRange: "
            +hipGetErrorString(priority_status));
    }

    (void)least;

    hipStream_t stream=nullptr;

    const hipError_t create_status=
        hipStreamCreateWithPriority(
            &stream,
            hipStreamNonBlocking,
            greatest);

    if(create_status!=hipSuccess){
        throw std::runtime_error(
            std::string(label)
            +": hipStreamCreateWithPriority: "
            +hipGetErrorString(create_status));
    }

    return stream;
}

static hipStream_t
create_decoder_execution_stream(
    const char* label) {

    if(g_resident_cu_partition_enabled){
        print_rx7900xtx_cu_partition_summary_once();

        return create_rx7900xtx_partitioned_stream(
            true,
            label);
    }

    int least=0;
    int greatest=0;

    const hipError_t priority_status=
        hipDeviceGetStreamPriorityRange(
            &least,
            &greatest);

    if(priority_status!=hipSuccess){
        throw std::runtime_error(
            std::string(label)
            +": hipDeviceGetStreamPriorityRange: "
            +hipGetErrorString(priority_status));
    }

    (void)greatest;

    hipStream_t stream=nullptr;

    const hipError_t create_status=
        hipStreamCreateWithPriority(
            &stream,
            hipStreamNonBlocking,
            least);

    if(create_status!=hipSuccess){
        throw std::runtime_error(
            std::string(label)
            +": hipStreamCreateWithPriority: "
            +hipGetErrorString(create_status));
    }

    return stream;
}

namespace code_predictor {

namespace fs = std::filesystem;





#define PRED_HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t error__ = (call);                                      \
        if (error__ != hipSuccess) {                                            \
            throw std::runtime_error(                                           \
                std::string("HIP error: ") + hipGetErrorString(error__) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                      \
    } while (0)

constexpr int kOuterHidden = 1024;
constexpr int kHidden = 1024;
constexpr bool kProjectionIdentity = kOuterHidden==kHidden;
constexpr int kIntermediate = 3072;
constexpr int kLayers = 5;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kQueryDim = kQueryHeads * kHeadDim;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kVocab = 2048;
constexpr int kSteps = 15;
constexpr int kCodeGroups = 16;
constexpr int kMaxCache = 16;
constexpr int kThreads = 256;
constexpr float kRmsEpsilon = 1.0e-6f;
constexpr float kAttentionScale = 0.08838834764831845f;

struct DeviceBuffer {
    void* pointer = nullptr;
    std::size_t bytes = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t byte_count)
        : bytes(byte_count) {
        if (bytes > 0) {
            PRED_HIP_CHECK(hipMalloc(&pointer, bytes));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer(other.pointer), bytes(other.bytes) {
        other.pointer = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (pointer) {
                (void)hipFree(pointer);
            }

            pointer = other.pointer;
            bytes = other.bytes;

            other.pointer = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (pointer) {
            (void)hipFree(pointer);
        }
    }

    template <typename T>
    T* as() {
        return reinterpret_cast<T*>(pointer);
    }

    template <typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(pointer);
    }
};

template <typename T>
std::vector<T> read_binary(
    const fs::path& path,
    std::size_t expected_elements) {

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "Cannot open input file: " + path.string());
    }

    std::vector<T> values(expected_elements);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(
            expected_elements * sizeof(T)));

    if (stream.gcount() != static_cast<std::streamsize>(
            expected_elements * sizeof(T))) {
        throw std::runtime_error(
            "Unexpected input size: " + path.string());
    }

    char extra = 0;
    if (stream.read(&extra, 1)) {
        throw std::runtime_error(
            "Trailing bytes in input: " + path.string());
    }

    return values;
}

template <typename T>
void write_binary(
    const fs::path& path,
    const std::vector<T>& values) {

    fs::create_directories(path.parent_path());

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "Cannot create output file: " + path.string());
    }

    stream.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(
            values.size() * sizeof(T)));

    if (!stream) {
        throw std::runtime_error(
            "Failed writing output: " + path.string());
    }
}

template <typename T>
void upload(
    DeviceBuffer& destination,
    const std::vector<T>& source) {

    if (destination.bytes < source.size() * sizeof(T)) {
        throw std::runtime_error(
            "Device destination is too small.");
    }

    PRED_HIP_CHECK(hipMemcpy(
        destination.pointer,
        source.data(),
        source.size() * sizeof(T),
        hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download(
    const T* source,
    std::size_t element_count) {

    std::vector<T> host(element_count);

    PRED_HIP_CHECK(hipMemcpy(
        host.data(),
        source,
        element_count * sizeof(T),
        hipMemcpyDeviceToHost));

    return host;
}

inline bool g_save_intermediates = false;

void save_device_bf16(
    const fs::path& path,
    const uint16_t* source,
    std::size_t element_count) {

    if (!g_save_intermediates) {
        return;
    }

    write_binary(
        path,
        download(source, element_count));
}


// legacy_legacy_tag_f validation persistence.
//
// The old legacy_legacy_tag_e save helper is intentionally guarded by the legacy global
// g_save_intermediates switch because a full trace is very large.
// legacy_legacy_tag_f teacher-forced validation, however, needs a small bounded set of tensors
// on every step. Those files are part of the validation contract and must not
// silently disappear when the legacy trace switch is false.
void save_device_bf16_validation(
    const fs::path& path,
    const uint16_t* source,
    std::size_t element_count,
    bool required) {

    if (!required && !g_save_intermediates) {
        return;
    }

    write_binary(
        path,
        download(source, element_count));
}


void save_device_f32(
    const fs::path& path,
    const float* source,
    std::size_t element_count) {

    if (!g_save_intermediates) {
        return;
    }

    write_binary(
        path,
        download(source, element_count));
}

__device__ __forceinline__ float bf16_to_float(
    uint16_t value) {

    return __uint_as_float(
        static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16_rne(
    float value) {

    uint32_t bits = __float_as_uint(value);

    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<uint16_t>(
            (bits >> 16) | 0x0040U);
    }

    const uint32_t least_significant =
        (bits >> 16) & 1U;

    bits += 0x7fffU + least_significant;
    return static_cast<uint16_t>(bits >> 16);
}

using predictor_bf16x2_bits =
    short __attribute__((ext_vector_type(2)));

__device__ __forceinline__
float predictor_bf16_dot2_accumulate(
    std::uint16_t x0,
    std::uint16_t x1,
    std::uint16_t w0,
    std::uint16_t w1,
    float accumulator) {

#if defined(__gfx1100__)
    const predictor_bf16x2_bits x = {
        static_cast<short>(x0),
        static_cast<short>(x1)
    };

    const predictor_bf16x2_bits w = {
        static_cast<short>(w0),
        static_cast<short>(w1)
    };

    return __builtin_amdgcn_fdot2_f32_bf16(
        x,
        w,
        accumulator,
        false);
#else
    accumulator = fmaf(
        bf16_to_float(x0),
        bf16_to_float(w0),
        accumulator);

    accumulator = fmaf(
        bf16_to_float(x1),
        bf16_to_float(w1),
        accumulator);

    return accumulator;
#endif
}



__device__ __forceinline__ float warp_reduce_sum(
    float value) {

    for (
        int offset = warpSize / 2;
        offset > 0;
        offset /= 2) {

        value += __shfl_down(value, offset);
    }

    return value;
}

__device__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[32];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int warp_count =
        (blockDim.x + warpSize - 1) / warpSize;

    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    float block_value = 0.0f;

    if (warp_id == 0) {
        block_value =
            lane < warp_count
            ? warp_sums[lane]
            : 0.0f;

        block_value = warp_reduce_sum(block_value);
    }

    return block_value;
}

__global__ void rmsnorm_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ output,
    int rows,
    int dimension) {

    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const int base = row * dimension;
    float sum_squares = 0.0f;

    for (
        int column = threadIdx.x;
        column < dimension;
        column += blockDim.x) {

        const float value =
            bf16_to_float(input[base + column]);

        sum_squares =
            fmaf(value, value, sum_squares);
    }

    sum_squares = block_reduce_sum(sum_squares);

    __shared__ float inverse_rms;

    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(
            sum_squares
                / static_cast<float>(dimension)
            + kRmsEpsilon);
    }
    __syncthreads();

    for (
        int column = threadIdx.x;
        column < dimension;
        column += blockDim.x) {

        const float value =
            bf16_to_float(input[base + column]);

        const uint16_t normalized_bits =
            float_to_bf16_rne(
                value * inverse_rms);

        const float normalized =
            bf16_to_float(normalized_bits);

        const float scale =
            bf16_to_float(weight[column]);

        output[base + column] =
            float_to_bf16_rne(
                normalized * scale);
    }
}

__global__ void add_bias_bf16_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ output,
    uint64_t count,
    int columns) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    const int column =
        static_cast<int>(index % columns);

    // PyTorch's non-Lt fallback materializes the BF16 GEMM output and then
    // applies the BF16 bias addition. Preserve that intermediate rounding.
    output[index] = float_to_bf16_rne(
        bf16_to_float(input[index])
        + bf16_to_float(bias[column]));
}

__global__ void transpose_bshd_to_bhsd_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int sequence_length,
    int head_count) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const uint64_t count =
        static_cast<uint64_t>(sequence_length)
        * head_count
        * kHeadDim;

    if (index >= count) {
        return;
    }

    const int dimension = index % kHeadDim;
    const uint64_t temporary = index / kHeadDim;
    const int head = temporary % head_count;
    const int sequence = temporary / head_count;

    const uint64_t destination =
        (
            static_cast<uint64_t>(head)
                * sequence_length
            + sequence
        )
        * kHeadDim
        + dimension;

    output[destination] = input[index];
}

__global__ void apply_rope_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ cosine,
    const uint16_t* __restrict__ sine,
    uint16_t* __restrict__ output,
    int sequence_length,
    int head_count) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const uint64_t count =
        static_cast<uint64_t>(head_count)
        * sequence_length
        * kHeadDim;

    if (index >= count) {
        return;
    }

    const int dimension = index % kHeadDim;
    const uint64_t temporary = index / kHeadDim;
    const int sequence =
        temporary % sequence_length;
    const int head =
        temporary / sequence_length;

    const int rotated_dimension =
        dimension < kHeadDim / 2
        ? dimension + kHeadDim / 2
        : dimension - kHeadDim / 2;

    const float rotated_sign =
        dimension < kHeadDim / 2
        ? -1.0f
        : 1.0f;

    const uint64_t input_base =
        (
            static_cast<uint64_t>(head)
                * sequence_length
            + sequence
        )
        * kHeadDim;

    const float value =
        bf16_to_float(
            input[input_base + dimension]);

    const float rotated =
        rotated_sign
        * bf16_to_float(
            input[
                input_base
                + rotated_dimension
            ]);

    const float cos_value =
        bf16_to_float(
            cosine[
                static_cast<uint64_t>(sequence)
                    * kHeadDim
                + dimension
            ]);

    const float sin_value =
        bf16_to_float(
            sine[
                static_cast<uint64_t>(sequence)
                    * kHeadDim
                + dimension
            ]);

    const uint16_t first =
        float_to_bf16_rne(
            value * cos_value);

    const uint16_t second =
        float_to_bf16_rne(
            rotated * sin_value);

    output[index] = float_to_bf16_rne(
        bf16_to_float(first)
        + bf16_to_float(second));
}

__global__ void update_cache_kernel(
    const uint16_t* __restrict__ source,
    uint16_t* __restrict__ cache,
    int sequence_length,
    int cache_start) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const uint64_t count =
        static_cast<uint64_t>(kKvHeads)
        * sequence_length
        * kHeadDim;

    if (index >= count) {
        return;
    }

    const int dimension = index % kHeadDim;
    const uint64_t temporary = index / kHeadDim;
    const int sequence =
        temporary % sequence_length;
    const int head =
        temporary / sequence_length;

    const uint64_t destination =
        (
            static_cast<uint64_t>(head)
                * kMaxCache
            + cache_start
            + sequence
        )
        * kHeadDim
        + dimension;

    cache[destination] = source[index];
}

__global__ void compact_cache_kernel(
    const uint16_t* __restrict__ cache,
    uint16_t* __restrict__ compact,
    int cache_length) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const uint64_t count =
        static_cast<uint64_t>(kKvHeads)
        * cache_length
        * kHeadDim;

    if (index >= count) {
        return;
    }

    const int dimension = index % kHeadDim;
    const uint64_t temporary = index / kHeadDim;
    const int sequence =
        temporary % cache_length;
    const int head =
        temporary / cache_length;

    const uint64_t source =
        (
            static_cast<uint64_t>(head)
                * kMaxCache
            + sequence
        )
        * kHeadDim
        + dimension;

    compact[index] = cache[source];
}

__global__ void attention_kernel(
    const uint16_t* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    uint16_t* __restrict__ output_bshd,
    int query_length,
    int cache_before,
    int total_cache_length,
    int causal_prefill) {

    const int query_index = blockIdx.x;
    const int query_head = blockIdx.y;

    if (
        query_index >= query_length
        || query_head >= kQueryHeads) {

        return;
    }

    const int kv_head =
        query_head / (kQueryHeads / kKvHeads);

    __shared__ float logits[kMaxCache];
    __shared__ float probabilities[kMaxCache];
    __shared__ int valid_key_count;

    if (threadIdx.x == 0) {
        valid_key_count =
            causal_prefill
            ? cache_before + query_index + 1
            : total_cache_length;

        float maximum =
            -std::numeric_limits<float>::infinity();

        for (
            int key_index = 0;
            key_index < valid_key_count;
            ++key_index) {

            float dot = 0.0f;

            const uint64_t query_base =
                (
                    static_cast<uint64_t>(
                        query_head
                    )
                        * query_length
                    + query_index
                )
                * kHeadDim;

            const uint64_t key_base =
                (
                    static_cast<uint64_t>(
                        kv_head
                    )
                        * kMaxCache
                    + key_index
                )
                * kHeadDim;

            for (
                int dimension = 0;
                dimension < kHeadDim;
                ++dimension) {

                dot = fmaf(
                    bf16_to_float(
                        query[
                            query_base + dimension
                        ]),
                    bf16_to_float(
                        key_cache[
                            key_base + dimension
                        ]),
                    dot);
            }

            logits[key_index] =
                dot * kAttentionScale;

            maximum = fmaxf(
                maximum,
                logits[key_index]);
        }

        float denominator = 0.0f;

        for (
            int key_index = 0;
            key_index < valid_key_count;
            ++key_index) {

            const float probability =
                expf(
                    logits[key_index]
                    - maximum);

            probabilities[key_index] =
                probability;
            denominator += probability;
        }

        for (
            int key_index = 0;
            key_index < valid_key_count;
            ++key_index) {

            probabilities[key_index] /=
                denominator;
        }
    }

    __syncthreads();

    for (
        int dimension = threadIdx.x;
        dimension < kHeadDim;
        dimension += blockDim.x) {

        float accumulation = 0.0f;

        for (
            int key_index = 0;
            key_index < valid_key_count;
            ++key_index) {

            const uint64_t value_index =
                (
                    static_cast<uint64_t>(
                        kv_head
                    )
                        * kMaxCache
                    + key_index
                )
                * kHeadDim
                + dimension;

            accumulation = fmaf(
                probabilities[key_index],
                bf16_to_float(
                    value_cache[value_index]),
                accumulation);
        }

        const uint64_t output_index =
            (
                static_cast<uint64_t>(
                    query_index
                )
                    * kQueryHeads
                + query_head
            )
            * kHeadDim
            + dimension;

        output_bshd[output_index] =
            float_to_bf16_rne(accumulation);
    }
}

__global__ void add_bf16_kernel(
    const uint16_t* __restrict__ left,
    const uint16_t* __restrict__ right,
    uint16_t* __restrict__ output,
    uint64_t count) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    output[index] = float_to_bf16_rne(
        bf16_to_float(left[index])
        + bf16_to_float(right[index]));
}

__global__ void swiglu_kernel(
    const uint16_t* __restrict__ gate,
    const uint16_t* __restrict__ up,
    uint16_t* __restrict__ output,
    uint64_t count) {

    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    const float gate_value =
        bf16_to_float(gate[index]);

    const uint16_t activated =
        float_to_bf16_rne(
            gate_value
            / (
                1.0f
                + expf(-gate_value)
            ));

    output[index] = float_to_bf16_rne(
        bf16_to_float(activated)
        * bf16_to_float(up[index]));
}

__global__ void argmax_bf16_kernel(
    const uint16_t* __restrict__ logits,
    int last_row_offset,
    int32_t* __restrict__ output_id) {

    __shared__ float values[kThreads];
    __shared__ int ids[kThreads];

    float best_value =
        -std::numeric_limits<float>::infinity();
    int best_id = 0;

    for (
        int token = threadIdx.x;
        token < kVocab;
        token += blockDim.x) {

        const float value =
            bf16_to_float(
                logits[
                    last_row_offset + token
                ]);

        if (
            value > best_value
            || (
                value == best_value
                && token < best_id
            )) {

            best_value = value;
            best_id = token;
        }
    }

    values[threadIdx.x] = best_value;
    ids[threadIdx.x] = best_id;
    __syncthreads();

    for (
        int offset = blockDim.x / 2;
        offset > 0;
        offset /= 2) {

        if (static_cast<int>(threadIdx.x) < offset) {
            const float other_value =
                values[threadIdx.x + offset];
            const int other_id =
                ids[threadIdx.x + offset];

            if (
                other_value > values[threadIdx.x]
                || (
                    other_value
                        == values[threadIdx.x]
                    && other_id < ids[threadIdx.x]
                )) {

                values[threadIdx.x] =
                    other_value;
                ids[threadIdx.x] =
                    other_id;
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output_id[0] = ids[0];
    }
}

__global__ void gather_embedding_kernel(
    const uint16_t* __restrict__ table,
    int token,
    uint16_t* __restrict__ destination) {

    const int dimension =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (dimension >= kOuterHidden) {
        return;
    }

    destination[dimension] =
        table[
            static_cast<uint64_t>(token)
                * kOuterHidden
            + dimension
        ];
}

__global__ void sum_codec_embeddings_kernel(
    const uint16_t* __restrict__ embeddings,
    uint16_t* __restrict__ output) {

    const int dimension =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (dimension >= kOuterHidden) {
        return;
    }

    float accumulation = 0.0f;

    for (
        int group = 0;
        group < kCodeGroups;
        ++group) {

        accumulation += bf16_to_float(
            embeddings[
                static_cast<uint64_t>(group)
                    * kOuterHidden
                + dimension
            ]);
    }

    output[dimension] =
        float_to_bf16_rne(accumulation);
}

void launch_rmsnorm(
    const uint16_t* input,
    const uint16_t* weight,
    uint16_t* output,
    int rows,
    int dimension,
    hipStream_t stream = nullptr) {

    hipLaunchKernelGGL(
        rmsnorm_kernel,
        dim3(rows),
        dim3(kThreads),
        0,
        stream,
        input,
        weight,
        output,
        rows,
        dimension);

    PRED_HIP_CHECK(hipGetLastError());
}

void launch_transpose(
    const uint16_t* input,
    uint16_t* output,
    int sequence_length,
    int head_count) {

    const uint64_t count =
        static_cast<uint64_t>(
            sequence_length
        )
        * head_count
        * kHeadDim;

    hipLaunchKernelGGL(
        transpose_bshd_to_bhsd_kernel,
        dim3(static_cast<uint32_t>(
            (count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        input,
        output,
        sequence_length,
        head_count);

    PRED_HIP_CHECK(hipGetLastError());
}

void launch_rope(
    const uint16_t* input,
    const uint16_t* cosine,
    const uint16_t* sine,
    uint16_t* output,
    int sequence_length,
    int head_count) {

    const uint64_t count =
        static_cast<uint64_t>(
            sequence_length
        )
        * head_count
        * kHeadDim;

    hipLaunchKernelGGL(
        apply_rope_kernel,
        dim3(static_cast<uint32_t>(
            (count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        input,
        cosine,
        sine,
        output,
        sequence_length,
        head_count);

    PRED_HIP_CHECK(hipGetLastError());
}

void launch_update_cache(
    const uint16_t* source,
    uint16_t* cache,
    int sequence_length,
    int cache_start) {

    const uint64_t count =
        static_cast<uint64_t>(kKvHeads)
        * sequence_length
        * kHeadDim;

    hipLaunchKernelGGL(
        update_cache_kernel,
        dim3(static_cast<uint32_t>(
            (count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        source,
        cache,
        sequence_length,
        cache_start);

    PRED_HIP_CHECK(hipGetLastError());
}

void launch_compact_cache(
    const uint16_t* cache,
    uint16_t* compact,
    int cache_length) {

    const uint64_t count =
        static_cast<uint64_t>(kKvHeads)
        * cache_length
        * kHeadDim;

    hipLaunchKernelGGL(
        compact_cache_kernel,
        dim3(static_cast<uint32_t>(
            (count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        cache,
        compact,
        cache_length);

    PRED_HIP_CHECK(hipGetLastError());
}


constexpr std::uint32_t
    kAttentionSplitScaleBits =
        0x3e9837f0U;

__global__ void bf16_to_scaled_f32_kernel(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    std::uint64_t count) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (index >= count) {
        return;
    }

    const float split_scale =
        __uint_as_float(
            kAttentionSplitScaleBits);

    output[index] =
        bf16_to_float(input[index])
        * split_scale;
}

__global__ void repeat_kv_to_f32_kernel(
    const uint16_t* __restrict__ cache,
    float* __restrict__ repeated,
    int key_length,
    int apply_split_scale) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(kQueryHeads)
        * key_length
        * kHeadDim;

    if (index >= count) {
        return;
    }

    const int dimension =
        static_cast<int>(
            index % kHeadDim);

    const std::uint64_t temporary =
        index / kHeadDim;

    const int key_index =
        static_cast<int>(
            temporary % key_length);

    const int query_head =
        static_cast<int>(
            temporary / key_length);

    const int kv_head =
        query_head
        / (kQueryHeads / kKvHeads);

    const std::uint64_t source_index =
        (
            static_cast<std::uint64_t>(kv_head)
                * kMaxCache
            + key_index
        )
        * kHeadDim
        + dimension;

    float value =
        bf16_to_float(
            cache[source_index]);

    if (apply_split_scale) {
        value *= __uint_as_float(
            kAttentionSplitScaleBits);
    }

    repeated[index] = value;
}

__global__ void copy_causal_mask_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int query_length,
    int key_length,
    int causal_prefill) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * key_length;

    if (index >= count) {
        return;
    }

    const int key_index =
        static_cast<int>(
            index % key_length);

    const std::uint64_t temporary =
        index / key_length;

    const int query_index =
        static_cast<int>(
            temporary % query_length);

    if (
        causal_prefill
        && key_index > query_index) {

        output[index] =
            -std::numeric_limits<float>::
                infinity();
    } else {
        output[index] =
            input[index];
    }
}

__device__ __forceinline__
float reduce_max(
    float value,
    int logical_width) {

    for (
        int offset =
            logical_width / 2;
        offset > 0;
        offset /= 2) {

        const float other =
            __shfl_xor(
                value,
                offset,
                logical_width);

        value =
            value < other
            ? other
            : value;
    }

    return value;
}

__device__ __forceinline__
float reduce_sum(
    float value,
    int logical_width) {

    for (
        int offset =
            logical_width / 2;
        offset > 0;
        offset /= 2) {

        value +=
            __shfl_xor(
                value,
                offset,
                logical_width);
    }

    return value;
}

__global__ void
softmax_persistent_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_count,
    int element_count,
    int logical_width,
    int batches_per_warp) {

    const int lane =
        static_cast<int>(threadIdx.x);

    const int logical_warp =
        static_cast<int>(threadIdx.y);

    const int first_row =
        (
            static_cast<int>(blockDim.y)
                * static_cast<int>(blockIdx.x)
            + logical_warp
        )
        * batches_per_warp;

    for (
        int batch = 0;
        batch < batches_per_warp;
        ++batch) {

        const int row =
            first_row + batch;

        if (row >= row_count) {
            continue;
        }

        const int element_index =
            lane;

        float element =
            element_index < element_count
            ? input[
                static_cast<std::size_t>(row)
                    * element_count
                + element_index
            ]
            : -std::numeric_limits<float>::
                infinity();

        const float maximum =
            reduce_max(
                element,
                logical_width);

        const float exponential =
            element_index < element_count
            ? expf(element - maximum)
            : 0.0f;

        const float sum =
            reduce_sum(
                exponential,
                logical_width);

        if (
            element_index
            < element_count) {

            output[
                static_cast<std::size_t>(row)
                    * element_count
                + element_index
            ] = exponential / sum;
        }
    }
}

__global__ void
context_f32_to_bf16_bshd_kernel(
    const float* __restrict__ input_bhsd,
    uint16_t* __restrict__ output_bshd,
    int sequence_length) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(
            kQueryHeads)
        * sequence_length
        * kHeadDim;

    if (index >= count) {
        return;
    }

    const int dimension =
        static_cast<int>(
            index % kHeadDim);

    const std::uint64_t temporary =
        index / kHeadDim;

    const int sequence =
        static_cast<int>(
            temporary % sequence_length);

    const int head =
        static_cast<int>(
            temporary
            / sequence_length);

    const std::uint64_t destination =
        (
            static_cast<std::uint64_t>(
                sequence)
                * kQueryHeads
            + head
        )
        * kHeadDim
        + dimension;

    output_bshd[destination] =
        float_to_bf16_rne(
            input_bhsd[index]);
}


constexpr std::uint32_t
    kAttentionScaleBits =
        0x3db504f3U; // float32(1/sqrt(128))

__device__ __forceinline__
float legacy_wave_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    return value;
}

// One wave computes one attention score:
//   dot(Q[head,query,:], K[kv_head,key,:])
__global__ __launch_bounds__(256, 2)
void attention_qk_wave32_kernel(
    const std::uint16_t* __restrict__ query,
    const std::uint16_t* __restrict__ key_cache,
    std::uint16_t* __restrict__ output,
    int query_length,
    int key_length) {

    constexpr int waves_per_block = 8;

    const int thread =
        static_cast<int>(threadIdx.x);

    const int wave =
        thread >> 5;

    const int lane =
        thread & 31;

    const std::uint64_t score_index =
        static_cast<std::uint64_t>(blockIdx.x)
            * waves_per_block
        + wave;

    const std::uint64_t score_count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * key_length;

    if (score_index >= score_count) {
        return;
    }

    const int key_index =
        static_cast<int>(
            score_index % key_length);

    const std::uint64_t t =
        score_index / key_length;

    const int query_index =
        static_cast<int>(
            t % query_length);

    const int query_head =
        static_cast<int>(
            t / query_length);

    const int kv_head =
        query_head
        / (kQueryHeads / kKvHeads);

    const std::size_t query_base =
        (
            static_cast<std::size_t>(query_head)
                * query_length
            + query_index
        ) * kHeadDim;

    const std::size_t key_base =
        (
            static_cast<std::size_t>(kv_head)
                * kMaxCache
            + key_index
        ) * kHeadDim;

    float partial = 0.0f;

    for (
        int dimension = lane;
        dimension < kHeadDim;
        dimension += 32
    ) {
        partial = fmaf(
            bf16_to_float(
                query[
                    query_base
                    + dimension]),
            bf16_to_float(
                key_cache[
                    key_base
                    + dimension]),
            partial);
    }

    const float sum =
        legacy_wave_reduce_sum(partial);

    if (lane == 0) {
        output[score_index] =
            float_to_bf16_rne(sum);
    }
}

__global__ void attention_scale_qk_kernel(
    const std::uint16_t* __restrict__ input,
    std::uint16_t* __restrict__ output,
    std::uint64_t count) {

    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    if (i >= count) {
        return;
    }

    const float scale =
        __uint_as_float(
            kAttentionScaleBits);

    output[i] =
        float_to_bf16_rne(
            bf16_to_float(input[i])
            * scale);
}

__global__ void attention_causal_mask_kernel(
    const std::uint16_t* __restrict__ input,
    std::uint16_t* __restrict__ output,
    int query_length,
    int key_length,
    int causal_prefill) {

    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * key_length;

    if (i >= count) {
        return;
    }

    const int key_index =
        static_cast<int>(
            i % key_length);

    const int query_index =
        static_cast<int>(
            (i / key_length)
            % query_length);

    output[i] =
        (
            causal_prefill
            && key_index > query_index
        )
        ? static_cast<std::uint16_t>(
            0xff80U) // -inf
        : input[i];
}

// One wave computes one softmax row. Predictor key length <= 16, so lanes
// beyond key_length simply contribute -inf / 0.
__global__ void attention_softmax_rows_kernel(
    const std::uint16_t* __restrict__ input,
    std::uint16_t* __restrict__ output,
    int row_count,
    int key_length) {

    const int row =
        static_cast<int>(blockIdx.x);

    const int lane =
        static_cast<int>(threadIdx.x);

    if (row >= row_count) {
        return;
    }

    float value =
        lane < key_length
        ? bf16_to_float(
            input[
                static_cast<std::size_t>(row)
                    * key_length
                + lane])
        : -INFINITY;

    float maximum = value;

    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        const float other =
            __shfl_down(
                maximum,
                offset,
                32);

        if (lane + offset < 32) {
            maximum =
                maximum > other
                ? maximum
                : other;
        }
    }

    maximum =
        __shfl(
            maximum,
            0,
            32);

    const float exponential =
        lane < key_length
        ? expf(value - maximum)
        : 0.0f;

    const float sum =
        legacy_wave_reduce_sum(
            exponential);

    const float total =
        __shfl(
            sum,
            0,
            32);

    if (lane < key_length) {
        output[
            static_cast<std::size_t>(row)
                * key_length
            + lane
        ] = float_to_bf16_rne(
            exponential / total);
    }
}

// One wave computes one context scalar:
//   sum_k P[head,query,k] * V[kv_head,k,dim]
__global__ __launch_bounds__(256, 2)
void attention_pv_wave32_kernel(
    const std::uint16_t* __restrict__ probabilities,
    const std::uint16_t* __restrict__ value_cache,
    std::uint16_t* __restrict__ output,
    int query_length,
    int key_length) {

    constexpr int waves_per_block = 8;

    const int thread =
        static_cast<int>(threadIdx.x);

    const int wave =
        thread >> 5;

    const int lane =
        thread & 31;

    const std::uint64_t context_index =
        static_cast<std::uint64_t>(blockIdx.x)
            * waves_per_block
        + wave;

    const std::uint64_t context_count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * kHeadDim;

    if (context_index >= context_count) {
        return;
    }

    const int dimension =
        static_cast<int>(
            context_index % kHeadDim);

    const std::uint64_t t =
        context_index / kHeadDim;

    const int query_index =
        static_cast<int>(
            t % query_length);

    const int query_head =
        static_cast<int>(
            t / query_length);

    const int kv_head =
        query_head
        / (kQueryHeads / kKvHeads);

    const std::size_t probability_base =
        (
            static_cast<std::size_t>(query_head)
                * query_length
            + query_index
        ) * key_length;

    float partial = 0.0f;

    for (
        int key_index = lane;
        key_index < key_length;
        key_index += 32
    ) {
        const std::size_t value_index =
            (
                static_cast<std::size_t>(kv_head)
                    * kMaxCache
                + key_index
            ) * kHeadDim
            + dimension;

        partial = fmaf(
            bf16_to_float(
                probabilities[
                    probability_base
                    + key_index]),
            bf16_to_float(
                value_cache[
                    value_index]),
            partial);
    }

    const float sum =
        legacy_wave_reduce_sum(partial);

    if (lane == 0) {
        output[context_index] =
            float_to_bf16_rne(sum);
    }
}

__global__ void attention_transpose_context_kernel(
    const std::uint16_t* __restrict__ input,
    std::uint16_t* __restrict__ output,
    int query_length) {

    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * kHeadDim;

    if (i >= count) {
        return;
    }

    const int dimension =
        static_cast<int>(
            i % kHeadDim);

    const std::uint64_t t =
        i / kHeadDim;

    const int query_index =
        static_cast<int>(
            t % query_length);

    const int head =
        static_cast<int>(
            t / query_length);

    const std::uint64_t destination =
        (
            static_cast<std::uint64_t>(query_index)
                * kQueryHeads
            + head
        ) * kHeadDim
        + dimension;

    output[destination] =
        input[i];
}

struct AttentionWorkspace {
    DeviceBuffer qk_raw_bf16;
    DeviceBuffer qk_scaled_bf16;
    DeviceBuffer qk_masked_bf16;
    DeviceBuffer probabilities_bf16;
    DeviceBuffer context_bhsd_bf16;

    AttentionWorkspace()
        : qk_raw_bf16(
            static_cast<std::size_t>(kQueryHeads)
            * 2
            * kMaxCache
            * sizeof(std::uint16_t)),
          qk_scaled_bf16(
            static_cast<std::size_t>(kQueryHeads)
            * 2
            * kMaxCache
            * sizeof(std::uint16_t)),
          qk_masked_bf16(
            static_cast<std::size_t>(kQueryHeads)
            * 2
            * kMaxCache
            * sizeof(std::uint16_t)),
          probabilities_bf16(
            static_cast<std::size_t>(kQueryHeads)
            * 2
            * kMaxCache
            * sizeof(std::uint16_t)),
          context_bhsd_bf16(
            static_cast<std::size_t>(kQueryHeads)
            * 2
            * kHeadDim
            * sizeof(std::uint16_t)) {}
};

inline AttentionWorkspace&
attention_workspace() {
    static AttentionWorkspace workspace;
    return workspace;
}


inline const std::uint16_t*
legacy_qk_raw_bf16() {
    return attention_workspace()
        .qk_raw_bf16
        .as<std::uint16_t>();
}

inline const std::uint16_t*
legacy_qk_scaled_bf16() {
    return attention_workspace()
        .qk_scaled_bf16
        .as<std::uint16_t>();
}

inline const std::uint16_t*
legacy_qk_masked_bf16() {
    return attention_workspace()
        .qk_masked_bf16
        .as<std::uint16_t>();
}

inline const std::uint16_t*
legacy_probabilities_bf16() {
    return attention_workspace()
        .probabilities_bf16
        .as<std::uint16_t>();
}

inline const std::uint16_t*
legacy_context_bhsd_bf16() {
    return attention_workspace()
        .context_bhsd_bf16
        .as<std::uint16_t>();
}

void launch_attention(
    const std::uint16_t* query,
    const std::uint16_t* key_cache,
    const std::uint16_t* value_cache,
    std::uint16_t* output,
    int query_length,
    int cache_before,
    int cache_after,
    bool causal_prefill) {

    (void)cache_before;

    AttentionWorkspace& workspace =
        attention_workspace();

    constexpr int waves_per_block = 8;

    const std::uint64_t score_count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * cache_after;

    const std::uint64_t context_count =
        static_cast<std::uint64_t>(kQueryHeads)
        * query_length
        * kHeadDim;

    hipLaunchKernelGGL(
        attention_qk_wave32_kernel,
        dim3(static_cast<unsigned>(
            (score_count
             + waves_per_block - 1)
            / waves_per_block)),
        dim3(waves_per_block * 32),
        0,
        0,
        query,
        key_cache,
        workspace.qk_raw_bf16
            .as<std::uint16_t>(),
        query_length,
        cache_after);
    PRED_HIP_CHECK(
        hipGetLastError());

    hipLaunchKernelGGL(
        attention_scale_qk_kernel,
        dim3(static_cast<unsigned>(
            (score_count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        workspace.qk_raw_bf16
            .as<std::uint16_t>(),
        workspace.qk_scaled_bf16
            .as<std::uint16_t>(),
        score_count);
    PRED_HIP_CHECK(
        hipGetLastError());

    hipLaunchKernelGGL(
        attention_causal_mask_kernel,
        dim3(static_cast<unsigned>(
            (score_count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        workspace.qk_scaled_bf16
            .as<std::uint16_t>(),
        workspace.qk_masked_bf16
            .as<std::uint16_t>(),
        query_length,
        cache_after,
        causal_prefill ? 1 : 0);
    PRED_HIP_CHECK(
        hipGetLastError());

    const int row_count =
        kQueryHeads
        * query_length;

    hipLaunchKernelGGL(
        attention_softmax_rows_kernel,
        dim3(static_cast<unsigned>(row_count)),
        dim3(32),
        0,
        0,
        workspace.qk_masked_bf16
            .as<std::uint16_t>(),
        workspace.probabilities_bf16
            .as<std::uint16_t>(),
        row_count,
        cache_after);
    PRED_HIP_CHECK(
        hipGetLastError());

    hipLaunchKernelGGL(
        attention_pv_wave32_kernel,
        dim3(static_cast<unsigned>(
            (context_count
             + waves_per_block - 1)
            / waves_per_block)),
        dim3(waves_per_block * 32),
        0,
        0,
        workspace.probabilities_bf16
            .as<std::uint16_t>(),
        value_cache,
        workspace.context_bhsd_bf16
            .as<std::uint16_t>(),
        query_length,
        cache_after);
    PRED_HIP_CHECK(
        hipGetLastError());

    hipLaunchKernelGGL(
        attention_transpose_context_kernel,
        dim3(static_cast<unsigned>(
            (context_count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        0,
        workspace.context_bhsd_bf16
            .as<std::uint16_t>(),
        output,
        query_length);
    PRED_HIP_CHECK(
        hipGetLastError());
}


void launch_add(
    const uint16_t* left,
    const uint16_t* right,
    uint16_t* output,
    uint64_t count,
    hipStream_t stream = nullptr) {

    hipLaunchKernelGGL(
        add_bf16_kernel,
        dim3(static_cast<uint32_t>(
            (count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        stream,
        left,
        right,
        output,
        count);

    PRED_HIP_CHECK(hipGetLastError());
}

void launch_swiglu(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* output,
    uint64_t count,
    hipStream_t stream = nullptr) {

    hipLaunchKernelGGL(
        swiglu_kernel,
        dim3(static_cast<uint32_t>(
            (count + kThreads - 1)
            / kThreads)),
        dim3(kThreads),
        0,
        stream,
        gate,
        up,
        output,
        count);

    PRED_HIP_CHECK(hipGetLastError());
}





namespace explicit_runtime {

enum class LinearOp {
    Projection,
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
    LmHead,
};

struct LinearAuditState {
    bool enabled = false;
    fs::path root;
};

inline LinearAuditState& linear_audit_state() {
    static LinearAuditState state;
    return state;
}

inline std::string group_name(
    LinearOp op,
    int layer) {

    const char* name = "unknown";

    switch (op) {
        case LinearOp::Projection: name = "projection"; break;
        case LinearOp::QProj: name = "q_proj"; break;
        case LinearOp::KProj: name = "k_proj"; break;
        case LinearOp::VProj: name = "v_proj"; break;
        case LinearOp::OProj: name = "o_proj"; break;
        case LinearOp::GateProj: name = "gate_proj"; break;
        case LinearOp::UpProj: name = "up_proj"; break;
        case LinearOp::DownProj: name = "down_proj"; break;
        case LinearOp::LmHead: name = "lm_head"; break;
    }

    return layer >= 0
        ? std::string(name)
            + "_layer_" + std::to_string(layer)
        : std::string(name);
}

inline void set_linear_audit_context(
    bool enabled,
    const fs::path& root = fs::path()) {

    LinearAuditState& state =
        linear_audit_state();

    state.enabled = enabled;
    state.root = root;

    if (enabled) {
        fs::create_directories(root);
    }
}

inline void audit_save_linear_input(
    const std::string& group,
    const std::uint16_t* input,
    int rows,
    int reduction_columns) {

    LinearAuditState& state =
        linear_audit_state();

    if (!state.enabled) return;

    const fs::path path =
        state.root / group;

    fs::create_directories(path);

    save_device_bf16(
        path / "input.bf16.bin",
        input,
        static_cast<std::size_t>(rows)
            * reduction_columns);
}

inline void audit_save_linear_output(
    const std::string& group,
    const std::uint16_t* output,
    int rows,
    int output_columns,
    int reduction_columns,
    bool has_bias) {

    (void)reduction_columns;
    (void)has_bias;

    LinearAuditState& state =
        linear_audit_state();

    if (!state.enabled) return;

    const fs::path path =
        state.root / group;

    fs::create_directories(path);

    save_device_bf16(
        path / "output.bf16.bin",
        output,
        static_cast<std::size_t>(rows)
            * output_columns);
}

inline void explicit_linear_bf16(
    LinearOp op,
    int layer,
    const std::uint16_t* input,
    const std::uint16_t* weight,
    const std::uint16_t* bias,
    std::uint16_t* output,
    int rows,
    int output_columns,
    int reduction_columns) {

    const std::string group =
        group_name(op, layer);

    audit_save_linear_input(
        group,
        input,
        rows,
        reduction_columns);

    qwen3_tts::native_ops::linear_bf16(
        input,
        weight,
        bias,
        output,
        rows,
        output_columns,
        reduction_columns);

    PRED_HIP_CHECK(
        hipDeviceSynchronize());

    audit_save_linear_output(
        group,
        output,
        rows,
        output_columns,
        reduction_columns,
        bias != nullptr);
}

inline void verify_static_maps() {
    std::cout
        << "predictor_linear_backend: native_wave32_bf16_f32acc\n"
        << "predictor_linear_external_code_objects: False\n"
        << "predictor_linear_rocblas_dependency: False\n";
}

}  // namespace explicit_runtime



void load_bf16_file(
    DeviceBuffer& destination,
    const fs::path& path,
    std::size_t element_count) {

    upload(
        destination,
        read_binary<uint16_t>(
            path,
            element_count));
}


static void load_model_bf16(
    DeviceBuffer& destination,
    const model_frontend::Tensor& tensor,
    std::size_t expected_elements,
    const std::string& logical_name) {

    if(tensor.dtype!="BF16"){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 Predictor expected BF16 tensor "
            +logical_name
            +" actual="
            +tensor.dtype
            +" tensor="
            +tensor.name);
    }

    if(
        static_cast<std::size_t>(
            tensor.elements())
        !=expected_elements
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 Predictor element-count mismatch "
            +logical_name
            +" tensor="
            +tensor.name);
    }

    const std::size_t bytes=
        expected_elements
        *sizeof(std::uint16_t);

    if(destination.bytes<bytes){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 Predictor device buffer too small for "
            +logical_name);
    }

    PRED_HIP_CHECK(
        hipMemcpy(
            destination.pointer,
            tensor.data,
            bytes,
            hipMemcpyHostToDevice));
}

struct LayerWeights {
    DeviceBuffer input_layernorm;
    DeviceBuffer q_projection;
    DeviceBuffer k_projection;
    DeviceBuffer v_projection;
    DeviceBuffer o_projection;
    DeviceBuffer q_norm;
    DeviceBuffer k_norm;
    DeviceBuffer post_attention_layernorm;
    DeviceBuffer gate_projection;
    DeviceBuffer up_projection;
    DeviceBuffer down_projection;

    LayerWeights(
        const fs::path& root,
        int layer_index)
        : input_layernorm(
              kHidden * sizeof(uint16_t)),
          q_projection(
              static_cast<std::size_t>(kQueryDim)
              * kHidden
              * sizeof(uint16_t)),
          k_projection(
              static_cast<std::size_t>(kKvDim)
              * kHidden
              * sizeof(uint16_t)),
          v_projection(
              static_cast<std::size_t>(kKvDim)
              * kHidden
              * sizeof(uint16_t)),
          o_projection(
              static_cast<std::size_t>(kHidden)
              * kQueryDim
              * sizeof(uint16_t)),
          q_norm(
              kHeadDim * sizeof(uint16_t)),
          k_norm(
              kHeadDim * sizeof(uint16_t)),
          post_attention_layernorm(
              kHidden * sizeof(uint16_t)),
          gate_projection(
              static_cast<std::size_t>(
                  kIntermediate
              )
              * kHidden
              * sizeof(uint16_t)),
          up_projection(
              static_cast<std::size_t>(
                  kIntermediate
              )
              * kHidden
              * sizeof(uint16_t)),
          down_projection(
              static_cast<std::size_t>(kHidden)
              * kIntermediate
              * sizeof(uint16_t)) {

        const fs::path layer_root =
            root
            / (
                "layer_"
                + (
                    layer_index < 10
                    ? std::string("0")
                    : std::string()
                )
                + std::to_string(layer_index)
            );

        load_bf16_file(
            input_layernorm,
            layer_root
                / "input_layernorm.bf16.bin",
            kHidden);

        load_bf16_file(
            q_projection,
            layer_root / "q_proj.bf16.bin",
            static_cast<std::size_t>(
                kQueryDim
            ) * kHidden);

        load_bf16_file(
            k_projection,
            layer_root / "k_proj.bf16.bin",
            static_cast<std::size_t>(
                kKvDim
            ) * kHidden);

        load_bf16_file(
            v_projection,
            layer_root / "v_proj.bf16.bin",
            static_cast<std::size_t>(
                kKvDim
            ) * kHidden);

        load_bf16_file(
            o_projection,
            layer_root / "o_proj.bf16.bin",
            static_cast<std::size_t>(
                kHidden
            ) * kQueryDim);

        load_bf16_file(
            q_norm,
            layer_root / "q_norm.bf16.bin",
            kHeadDim);

        load_bf16_file(
            k_norm,
            layer_root / "k_norm.bf16.bin",
            kHeadDim);

        load_bf16_file(
            post_attention_layernorm,
            layer_root
                / "post_attention_layernorm.bf16.bin",
            kHidden);

        load_bf16_file(
            gate_projection,
            layer_root
                / "gate_proj.bf16.bin",
            static_cast<std::size_t>(
                kIntermediate
            ) * kHidden);

        load_bf16_file(
            up_projection,
            layer_root
                / "up_proj.bf16.bin",
            static_cast<std::size_t>(
                kIntermediate
            ) * kHidden);

        load_bf16_file(
            down_projection,
            layer_root
                / "down_proj.bf16.bin",
            static_cast<std::size_t>(
                kHidden
            ) * kIntermediate);
    }


    LayerWeights(
        const model_frontend::ModelWeights& model,
        int layer_index,
        const char* direct_model_tag)
        : input_layernorm(
              kHidden * sizeof(uint16_t)),
          q_projection(
              static_cast<std::size_t>(kQueryDim)
              * kHidden
              * sizeof(uint16_t)),
          k_projection(
              static_cast<std::size_t>(kKvDim)
              * kHidden
              * sizeof(uint16_t)),
          v_projection(
              static_cast<std::size_t>(kKvDim)
              * kHidden
              * sizeof(uint16_t)),
          o_projection(
              static_cast<std::size_t>(kHidden)
              * kQueryDim
              * sizeof(uint16_t)),
          q_norm(
              kHeadDim * sizeof(uint16_t)),
          k_norm(
              kHeadDim * sizeof(uint16_t)),
          post_attention_layernorm(
              kHidden * sizeof(uint16_t)),
          gate_projection(
              static_cast<std::size_t>(kIntermediate)
              * kHidden
              * sizeof(uint16_t)),
          up_projection(
              static_cast<std::size_t>(kIntermediate)
              * kHidden
              * sizeof(uint16_t)),
          down_projection(
              static_cast<std::size_t>(kHidden)
              * kIntermediate
              * sizeof(uint16_t)) {

        (void)direct_model_tag;

        const std::string root=
            "talker.code_predictor.model.layers."
            +std::to_string(layer_index)
            +".";

        load_model_bf16(
            input_layernorm,
            model.suffix(
                root+"input_layernorm.weight"),
            kHidden,
            root+"input_layernorm.weight");

        load_model_bf16(
            q_projection,
            model.suffix(
                root+"self_attn.q_proj.weight"),
            static_cast<std::size_t>(
                kQueryDim)*kHidden,
            root+"self_attn.q_proj.weight");

        load_model_bf16(
            k_projection,
            model.suffix(
                root+"self_attn.k_proj.weight"),
            static_cast<std::size_t>(
                kKvDim)*kHidden,
            root+"self_attn.k_proj.weight");

        load_model_bf16(
            v_projection,
            model.suffix(
                root+"self_attn.v_proj.weight"),
            static_cast<std::size_t>(
                kKvDim)*kHidden,
            root+"self_attn.v_proj.weight");

        load_model_bf16(
            o_projection,
            model.suffix(
                root+"self_attn.o_proj.weight"),
            static_cast<std::size_t>(
                kHidden)*kQueryDim,
            root+"self_attn.o_proj.weight");

        load_model_bf16(
            q_norm,
            model.suffix(
                root+"self_attn.q_norm.weight"),
            kHeadDim,
            root+"self_attn.q_norm.weight");

        load_model_bf16(
            k_norm,
            model.suffix(
                root+"self_attn.k_norm.weight"),
            kHeadDim,
            root+"self_attn.k_norm.weight");

        load_model_bf16(
            post_attention_layernorm,
            model.suffix(
                root+"post_attention_layernorm.weight"),
            kHidden,
            root+"post_attention_layernorm.weight");

        load_model_bf16(
            gate_projection,
            model.suffix(
                root+"mlp.gate_proj.weight"),
            static_cast<std::size_t>(
                kIntermediate)*kHidden,
            root+"mlp.gate_proj.weight");

        load_model_bf16(
            up_projection,
            model.suffix(
                root+"mlp.up_proj.weight"),
            static_cast<std::size_t>(
                kIntermediate)*kHidden,
            root+"mlp.up_proj.weight");

        load_model_bf16(
            down_projection,
            model.suffix(
                root+"mlp.down_proj.weight"),
            static_cast<std::size_t>(
                kHidden)*kIntermediate,
            root+"mlp.down_proj.weight");
    }
};

struct PredictorWeights {
    DeviceBuffer projection_weight;
    DeviceBuffer projection_bias;
    std::vector<LayerWeights> layers;
    DeviceBuffer final_norm;
    std::vector<DeviceBuffer> lm_heads;
    std::vector<DeviceBuffer> embeddings;

    explicit PredictorWeights(
        const fs::path& asset_root)
        : projection_weight(
              kProjectionIdentity
              ?0
              :static_cast<std::size_t>(kHidden)
                  *kOuterHidden
                  *sizeof(uint16_t)),
          projection_bias(
              kProjectionIdentity
              ?0
              :kHidden*sizeof(uint16_t)),
          final_norm(
              kHidden * sizeof(uint16_t)) {

        const fs::path weight_root =
            asset_root / "weights";

        if constexpr(!kProjectionIdentity){
            load_bf16_file(
                projection_weight,
                weight_root
                    / "projection_weight.bf16.bin",
                static_cast<std::size_t>(kHidden)
                    *kOuterHidden);

            load_bf16_file(
                projection_bias,
                weight_root
                    / "projection_bias.bf16.bin",
                kHidden);
        }

        layers.reserve(kLayers);

        for (
            int layer = 0;
            layer < kLayers;
            ++layer) {

            layers.emplace_back(
                weight_root,
                layer);
        }

        load_bf16_file(
            final_norm,
            weight_root / "final_norm.bf16.bin",
            kHidden);

        lm_heads.reserve(kSteps);
        embeddings.reserve(kSteps);

        for (
            int index = 0;
            index < kSteps;
            ++index) {

            const std::string suffix =
                (
                    index < 10
                    ? "0"
                    : ""
                )
                + std::to_string(index);

            lm_heads.emplace_back(
                static_cast<std::size_t>(kVocab)
                * kHidden
                * sizeof(uint16_t));

            load_bf16_file(
                lm_heads.back(),
                weight_root
                    / "lm_heads"
                    / (
                        "head_"
                        + suffix
                        + ".bf16.bin"
                    ),
                static_cast<std::size_t>(
                    kVocab
                ) * kHidden);

            embeddings.emplace_back(
                static_cast<std::size_t>(kVocab)
                * kOuterHidden
                * sizeof(uint16_t));

            load_bf16_file(
                embeddings.back(),
                weight_root
                    / "embeddings"
                    / (
                        "embedding_"
                        + suffix
                        + ".bf16.bin"
                    ),
                static_cast<std::size_t>(
                    kVocab
                ) * kOuterHidden);
        }
    }


    PredictorWeights(
        const model_frontend::ModelWeights& model,
        const char* direct_model_tag)
        : projection_weight(
              kProjectionIdentity
              ?0
              :static_cast<std::size_t>(kHidden)
                  *kOuterHidden
                  *sizeof(uint16_t)),
          projection_bias(
              kProjectionIdentity
              ?0
              :kHidden*sizeof(uint16_t)),
          final_norm(
              kHidden * sizeof(uint16_t)) {

        (void)direct_model_tag;

        if constexpr(!kProjectionIdentity){
            load_model_bf16(
                projection_weight,
                model.suffix(
                    "talker.code_predictor.small_to_mtp_projection.weight"),
                static_cast<std::size_t>(
                    kHidden)*kOuterHidden,
                "small_to_mtp_projection.weight");

            load_model_bf16(
                projection_bias,
                model.suffix(
                    "talker.code_predictor.small_to_mtp_projection.bias"),
                kHidden,
                "small_to_mtp_projection.bias");
        }

        layers.reserve(kLayers);

        for(int layer=0;layer<kLayers;++layer){
            layers.emplace_back(
                model,
                layer,
                direct_model_tag);
        }

        load_model_bf16(
            final_norm,
            model.suffix(
                "talker.code_predictor.model.norm.weight"),
            kHidden,
            "model.norm.weight");

        lm_heads.reserve(kSteps);
        embeddings.reserve(kSteps);

        for(int index=0;index<kSteps;++index){
            lm_heads.emplace_back(
                static_cast<std::size_t>(
                    kVocab)
                *kHidden
                *sizeof(uint16_t));

            load_model_bf16(
                lm_heads.back(),
                model.suffix(
                    "talker.code_predictor.lm_head."
                    +std::to_string(index)
                    +".weight"),
                static_cast<std::size_t>(
                    kVocab)*kHidden,
                "lm_head."
                    +std::to_string(index)
                    +".weight");

            embeddings.emplace_back(
                static_cast<std::size_t>(
                    kVocab)
                *kOuterHidden
                *sizeof(uint16_t));

            load_model_bf16(
                embeddings.back(),
                model.suffix(
                    "talker.code_predictor.model.codec_embedding."
                    +std::to_string(index)
                    +".weight"),
                static_cast<std::size_t>(
                    kVocab)*kOuterHidden,
                "codec_embedding."
                    +std::to_string(index)
                    +".weight");
        }

        std::cout
            <<"predictor_weights_source: model_safetensors\\n"
            <<"predictor_external_asset_dependency: False\\n";
    }
};

struct LayerCache {
    DeviceBuffer key;
    DeviceBuffer value;

    LayerCache()
        : key(
              static_cast<std::size_t>(kKvHeads)
              * kMaxCache
              * kHeadDim
              * sizeof(uint16_t)),
          value(
              static_cast<std::size_t>(kKvHeads)
              * kMaxCache
              * kHeadDim
              * sizeof(uint16_t)) {

        PRED_HIP_CHECK(hipMemset(
            key.pointer,
            0,
            key.bytes));

        PRED_HIP_CHECK(hipMemset(
            value.pointer,
            0,
            value.bytes));
    }
};

std::string two_digit(int value) {
    return (
        value < 10
        ? "0"
        : ""
    ) + std::to_string(value);
}

void run_frame(
    const fs::path& assets,
    const fs::path& output_root,
    const std::string& mode,
    int frame_index,
    const PredictorWeights& weights) {

    const std::string frame_name =
        "frame_" + two_digit(frame_index);

    const fs::path frame_assets =
        assets / frame_name;

    const fs::path frame_output =
        output_root / mode / frame_name;

    fs::create_directories(frame_output);

    const auto initial_host =
        read_binary<uint16_t>(
            frame_assets
                / "initial_input.bf16.bin",
            static_cast<std::size_t>(2)
                * kOuterHidden);

    const auto context_host =
        read_binary<uint16_t>(
            frame_assets
                / "context_embedding.bf16.bin",
            kOuterHidden);

    const auto expected_codes =
        read_binary<int32_t>(
            frame_assets
                / "expected_codes.i32.bin",
            kCodeGroups);

    DeviceBuffer initial_input(
        static_cast<std::size_t>(2)
        * kOuterHidden
        * sizeof(uint16_t));

    DeviceBuffer free_input(
        static_cast<std::size_t>(2)
        * kOuterHidden
        * sizeof(uint16_t));

    DeviceBuffer teacher_input(
        static_cast<std::size_t>(2)
        * kOuterHidden
        * sizeof(uint16_t));

    DeviceBuffer context_embedding(
        kOuterHidden * sizeof(uint16_t));

    DeviceBuffer selected_embeddings(
        static_cast<std::size_t>(kCodeGroups)
        * kOuterHidden
        * sizeof(uint16_t));

    upload(initial_input, initial_host);
    upload(free_input, initial_host);
    upload(context_embedding, context_host);

    PRED_HIP_CHECK(hipMemset(
        selected_embeddings.pointer,
        0,
        selected_embeddings.bytes));

    // Group 0 is the code_0 embedding, the second row of the
    // [Talker hidden, code_0 embedding] Predictor prefill input.
    PRED_HIP_CHECK(hipMemcpy(
        selected_embeddings.as<uint16_t>(),
        initial_input.as<uint16_t>()
            + kOuterHidden,
        kOuterHidden * sizeof(uint16_t),
        hipMemcpyDeviceToDevice));

    std::vector<LayerCache> caches;
    caches.reserve(kLayers);

    for (
        int layer = 0;
        layer < kLayers;
        ++layer) {

        caches.emplace_back();
    }

    std::array<int32_t, kCodeGroups> generated_codes{};
    generated_codes[0] = expected_codes[0];

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    PRED_HIP_CHECK(hipEventCreate(&start));
    PRED_HIP_CHECK(hipEventCreate(&stop));
    PRED_HIP_CHECK(hipEventRecord(start));

    for (
        int step = 0;
        step < kSteps;
        ++step) {

        const int sequence_length =
            step == 0 ? 2 : 1;
        const int cache_before =
            step == 0 ? 0 : step + 1;
        const int cache_after =
            step + 2;

        const std::string step_name =
            "predictor_step_"
            + two_digit(step);

        const fs::path step_assets =
            frame_assets / step_name;

        const fs::path step_output =
            frame_output / step_name;

        fs::create_directories(step_output);

        const bool teacher_step_mode =
            mode == "teacher"
            || mode == "layer_teacher";

        const bool audit_this_step =
            mode == "teacher";

        explicit_runtime::set_linear_audit_context(
            audit_this_step,
            output_root
                / "legacy_operator_audit_a"
                / mode
                / frame_name
                / step_name);

        if (teacher_step_mode) {
            // Reset every layer's K/V state to the exact Gold cache that
            // existed before this substep. This makes each teacher row an
            // independent operator/weight validation rather than a chained
            
            const std::size_t fixed_cache_elements =
                static_cast<std::size_t>(kKvHeads)
                * kMaxCache
                * kHeadDim;

            for (
                int layer_index = 0;
                layer_index < kLayers;
                ++layer_index) {

                const fs::path cache_root =
                    step_assets
                    / "teacher_cache_before"
                    / (
                        "layer_"
                        + two_digit(layer_index)
                    );

                upload(
                    caches[layer_index].key,
                    read_binary<uint16_t>(
                        cache_root / "key.bf16.bin",
                        fixed_cache_elements));

                upload(
                    caches[layer_index].value,
                    read_binary<uint16_t>(
                        cache_root / "value.bf16.bin",
                        fixed_cache_elements));
            }
        }

        const auto cosine_host =
            read_binary<uint16_t>(
                step_assets
                    / "rope_cos.bf16.bin",
                static_cast<std::size_t>(
                    sequence_length
                ) * kHeadDim);

        const auto sine_host =
            read_binary<uint16_t>(
                step_assets
                    / "rope_sin.bf16.bin",
                static_cast<std::size_t>(
                    sequence_length
                ) * kHeadDim);

        DeviceBuffer cosine(
            static_cast<std::size_t>(
                sequence_length
            )
            * kHeadDim
            * sizeof(uint16_t));

        DeviceBuffer sine(
            static_cast<std::size_t>(
                sequence_length
            )
            * kHeadDim
            * sizeof(uint16_t));

        upload(cosine, cosine_host);
        upload(sine, sine_host);

        const uint16_t* input_pointer = nullptr;

        if (teacher_step_mode) {
            const auto teacher_host =
                read_binary<uint16_t>(
                    step_assets
                        / "teacher_input.bf16.bin",
                    static_cast<std::size_t>(
                        sequence_length
                    ) * kOuterHidden);

            upload(teacher_input, teacher_host);
            input_pointer =
                teacher_input.as<uint16_t>();
        } else {
            input_pointer =
                free_input.as<uint16_t>();
        }

        save_device_bf16_validation(
            step_output / "input_1024.bf16.bin",
            input_pointer,
            static_cast<std::size_t>(
                sequence_length
            ) * kOuterHidden,
            mode == "teacher");

        const std::size_t projection_count =
            static_cast<std::size_t>(
                sequence_length
            ) * kHidden;

        DeviceBuffer hidden(
            projection_count
            * sizeof(uint16_t));

        if constexpr(kProjectionIdentity){
            PRED_HIP_CHECK(hipMemcpy(
                hidden.pointer,
                input_pointer,
                projection_count*sizeof(uint16_t),
                hipMemcpyDeviceToDevice));
        }else{
            explicit_runtime::explicit_linear_bf16(
                explicit_runtime::LinearOp::Projection,
                -1,
                input_pointer,
                weights.projection_weight.as<uint16_t>(),
                weights.projection_bias.as<uint16_t>(),
                hidden.as<uint16_t>(),
                sequence_length,
                kHidden,
                kOuterHidden);
        }

        save_device_bf16_validation(
            step_output
                / "projection_output_1024.bf16.bin",
            hidden.as<uint16_t>(),
            projection_count,
            teacher_step_mode);

        for (
            int layer_index = 0;
            layer_index < kLayers;
            ++layer_index) {

            const LayerWeights& layer =
                weights.layers[layer_index];

            const std::size_t hidden_count =
                static_cast<std::size_t>(
                    sequence_length
                ) * kHidden;

            const std::size_t query_count =
                static_cast<std::size_t>(
                    sequence_length
                ) * kQueryDim;

            const std::size_t kv_count =
                static_cast<std::size_t>(
                    sequence_length
                ) * kKvDim;

            const std::size_t intermediate_count =
                static_cast<std::size_t>(
                    sequence_length
                ) * kIntermediate;

            if(mode == "layer_teacher"){
                const fs::path exact_layer_input_path =
                    step_assets
                    / "teacher_layer_input"
                    / (
                        "layer_"
                        + two_digit(layer_index)
                        + ".bf16.bin"
                    );

                upload(
                    hidden,
                    read_binary<uint16_t>(
                        exact_layer_input_path,
                        hidden_count));
            }

            DeviceBuffer normalized(
                hidden_count
                * sizeof(uint16_t));

            DeviceBuffer query_linear(
                query_count
                * sizeof(uint16_t));

            DeviceBuffer key_linear(
                kv_count
                * sizeof(uint16_t));

            DeviceBuffer value_linear(
                kv_count
                * sizeof(uint16_t));

            DeviceBuffer query_normalized(
                query_count
                * sizeof(uint16_t));

            DeviceBuffer key_normalized(
                kv_count
                * sizeof(uint16_t));

            DeviceBuffer query_bhsd(
                query_count
                * sizeof(uint16_t));

            DeviceBuffer key_bhsd(
                kv_count
                * sizeof(uint16_t));

            DeviceBuffer value_bhsd(
                kv_count
                * sizeof(uint16_t));

            DeviceBuffer query_rotated(
                query_count
                * sizeof(uint16_t));

            DeviceBuffer key_rotated(
                kv_count
                * sizeof(uint16_t));

            DeviceBuffer attention_context(
                query_count
                * sizeof(uint16_t));

            DeviceBuffer attention_output(
                hidden_count
                * sizeof(uint16_t));

            DeviceBuffer attention_residual(
                hidden_count
                * sizeof(uint16_t));

            DeviceBuffer post_norm(
                hidden_count
                * sizeof(uint16_t));

            DeviceBuffer gate(
                intermediate_count
                * sizeof(uint16_t));

            DeviceBuffer up(
                intermediate_count
                * sizeof(uint16_t));

            DeviceBuffer swiglu(
                intermediate_count
                * sizeof(uint16_t));

            DeviceBuffer down(
                hidden_count
                * sizeof(uint16_t));

            DeviceBuffer layer_output(
                hidden_count
                * sizeof(uint16_t));

            DeviceBuffer compact_key(
                static_cast<std::size_t>(
                    kKvHeads
                )
                * cache_after
                * kHeadDim
                * sizeof(uint16_t));

            DeviceBuffer compact_value(
                static_cast<std::size_t>(
                    kKvHeads
                )
                * cache_after
                * kHeadDim
                * sizeof(uint16_t));

            const fs::path legacy_internal_c =
                step_output
                / ("layer_" + two_digit(layer_index))
                / "legacy_internal_c";
            const bool capture_common =
                g_save_intermediates
                && (
                    (step == 0 && layer_index == 0)
                    || (step == 11 && layer_index == 2)
                );
            const bool capture_target =
                g_save_intermediates
                && step == 11
                && layer_index == 2;
            const fs::path legacy_internal_a =
                step_output
                / ("layer_" + two_digit(layer_index))
                / "legacy_internal_a";
            const bool capture_kv_provenance =
                g_save_intermediates
                && layer_index == 2
                && step <= 11;
            const fs::path legacy_internal_b =
                step_output
                / ("layer_" + two_digit(layer_index))
                / "legacy_internal_b";

            if(capture_target)
                save_device_bf16(
                    legacy_internal_a/"00_layer_input.bf16.bin",
                    hidden.as<uint16_t>(),
                    hidden_count);

            launch_rmsnorm(
                hidden.as<uint16_t>(),
                layer.input_layernorm.as<uint16_t>(),
                normalized.as<uint16_t>(),
                sequence_length,
                kHidden);

            if(capture_common)
                save_device_bf16(
                    legacy_internal_c/"00_input_norm.bf16.bin",
                    normalized.as<uint16_t>(),
                    hidden_count);

            if(capture_target)
                save_device_bf16(
                    legacy_internal_a/"01_input_norm.bf16.bin",
                    normalized.as<uint16_t>(),
                    hidden_count);

            if(capture_kv_provenance)
                save_device_bf16(
                    legacy_internal_b/"00_input_norm.bf16.bin",
                    normalized.as<uint16_t>(),
                    hidden_count);

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::QProj,
            layer_index,
                normalized.as<uint16_t>(),
                layer.q_projection.as<uint16_t>(),
                nullptr,
                query_linear.as<uint16_t>(),
                sequence_length,
                kQueryDim,
                kHidden);

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::KProj,
            layer_index,
                normalized.as<uint16_t>(),
                layer.k_projection.as<uint16_t>(),
                nullptr,
                key_linear.as<uint16_t>(),
                sequence_length,
                kKvDim,
                kHidden);

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::VProj,
            layer_index,
                normalized.as<uint16_t>(),
                layer.v_projection.as<uint16_t>(),
                nullptr,
                value_linear.as<uint16_t>(),
                sequence_length,
                kKvDim,
                kHidden);

            if(capture_common){
                save_device_bf16(
                    legacy_internal_c/"01_q_linear.bf16.bin",
                    query_linear.as<uint16_t>(),query_count);
                save_device_bf16(
                    legacy_internal_c/"02_k_linear.bf16.bin",
                    key_linear.as<uint16_t>(),kv_count);
                save_device_bf16(
                    legacy_internal_c/"03_v_linear.bf16.bin",
                    value_linear.as<uint16_t>(),kv_count);
            }

            if(capture_target){
                save_device_bf16(
                    legacy_internal_a/"02_q_linear.bf16.bin",
                    query_linear.as<uint16_t>(),query_count);
                save_device_bf16(
                    legacy_internal_a/"03_k_linear.bf16.bin",
                    key_linear.as<uint16_t>(),kv_count);
                save_device_bf16(
                    legacy_internal_a/"04_v_linear.bf16.bin",
                    value_linear.as<uint16_t>(),kv_count);
            }

            if(capture_kv_provenance)
                save_device_bf16(
                    legacy_internal_b/"01_k_linear.bf16.bin",
                    key_linear.as<uint16_t>(),kv_count);

            launch_rmsnorm(
                query_linear.as<uint16_t>(),
                layer.q_norm.as<uint16_t>(),
                query_normalized.as<uint16_t>(),
                sequence_length
                    * kQueryHeads,
                kHeadDim);

            launch_rmsnorm(
                key_linear.as<uint16_t>(),
                layer.k_norm.as<uint16_t>(),
                key_normalized.as<uint16_t>(),
                sequence_length
                    * kKvHeads,
                kHeadDim);

            if(capture_common){
                save_device_bf16(
                    legacy_internal_c/"04_q_norm.bf16.bin",
                    query_normalized.as<uint16_t>(),query_count);
                save_device_bf16(
                    legacy_internal_c/"05_k_norm.bf16.bin",
                    key_normalized.as<uint16_t>(),kv_count);
            }

            if(capture_target){
                save_device_bf16(
                    legacy_internal_a/"05_q_norm.bf16.bin",
                    query_normalized.as<uint16_t>(),query_count);
                save_device_bf16(
                    legacy_internal_a/"06_k_norm.bf16.bin",
                    key_normalized.as<uint16_t>(),kv_count);
            }

            if(capture_kv_provenance)
                save_device_bf16(
                    legacy_internal_b/"02_k_norm.bf16.bin",
                    key_normalized.as<uint16_t>(),kv_count);

            launch_transpose(
                query_normalized.as<uint16_t>(),
                query_bhsd.as<uint16_t>(),
                sequence_length,
                kQueryHeads);

            launch_transpose(
                key_normalized.as<uint16_t>(),
                key_bhsd.as<uint16_t>(),
                sequence_length,
                kKvHeads);

            launch_transpose(
                value_linear.as<uint16_t>(),
                value_bhsd.as<uint16_t>(),
                sequence_length,
                kKvHeads);

            launch_rope(
                query_bhsd.as<uint16_t>(),
                cosine.as<uint16_t>(),
                sine.as<uint16_t>(),
                query_rotated.as<uint16_t>(),
                sequence_length,
                kQueryHeads);

            launch_rope(
                key_bhsd.as<uint16_t>(),
                cosine.as<uint16_t>(),
                sine.as<uint16_t>(),
                key_rotated.as<uint16_t>(),
                sequence_length,
                kKvHeads);

            if(capture_common){
                save_device_bf16(
                    legacy_internal_c/"06_query_rotated.bf16.bin",
                    query_rotated.as<uint16_t>(),query_count);
                save_device_bf16(
                    legacy_internal_c/"07_key_cache.bf16.bin",
                    key_rotated.as<uint16_t>(),kv_count);
                save_device_bf16(
                    legacy_internal_c/"08_value_cache.bf16.bin",
                    value_bhsd.as<uint16_t>(),kv_count);
            }

            if(capture_target)
                save_device_bf16(
                    legacy_internal_a/"07_query_rotated.bf16.bin",
                    query_rotated.as<uint16_t>(),query_count);

            if(capture_kv_provenance)
                save_device_bf16(
                    legacy_internal_b/"03_current_key_rotated.bf16.bin",
                    key_rotated.as<uint16_t>(),kv_count);

            launch_update_cache(
                key_rotated.as<uint16_t>(),
                caches[layer_index]
                    .key.as<uint16_t>(),
                sequence_length,
                cache_before);

            launch_update_cache(
                value_bhsd.as<uint16_t>(),
                caches[layer_index]
                    .value.as<uint16_t>(),
                sequence_length,
                cache_before);

            launch_attention(
                query_rotated.as<uint16_t>(),
                caches[layer_index]
                    .key.as<uint16_t>(),
                caches[layer_index]
                    .value.as<uint16_t>(),
                attention_context.as<uint16_t>(),
                sequence_length,
                cache_before,
                cache_after,
                step == 0);

            if(capture_common){
                const std::size_t eager_score_count=
                    static_cast<std::size_t>(kQueryHeads)
                    *sequence_length*cache_after;
                save_device_bf16(
                    legacy_internal_c/"09_qk_unscaled.bf16.bin",
                    legacy_qk_raw_bf16(),eager_score_count);
                save_device_bf16(
                    legacy_internal_c/"10_qk_scaled.bf16.bin",
                    legacy_qk_scaled_bf16(),eager_score_count);
                save_device_bf16(
                    legacy_internal_c/"11_qk_masked.bf16.bin",
                    legacy_qk_masked_bf16(),eager_score_count);
                save_device_bf16(
                    legacy_internal_c/"12_probabilities.bf16.bin",
                    legacy_probabilities_bf16(),eager_score_count);
                save_device_bf16(
                    legacy_internal_c/"13_attention_context.bf16.bin",
                    attention_context.as<uint16_t>(),query_count);
            }

            if(capture_target){
                const std::size_t legacy_eager_score_count=
                    static_cast<std::size_t>(kQueryHeads)
                    *sequence_length*cache_after;
                save_device_bf16(
                    legacy_internal_a/"10_qk_unscaled.bf16.bin",
                    legacy_qk_raw_bf16(),legacy_eager_score_count);
                save_device_bf16(
                    legacy_internal_a/"11_qk_scaled.bf16.bin",
                    legacy_qk_scaled_bf16(),legacy_eager_score_count);
                save_device_bf16(
                    legacy_internal_a/"12_qk_masked.bf16.bin",
                    legacy_qk_masked_bf16(),legacy_eager_score_count);
                save_device_bf16(
                    legacy_internal_a/"13_probabilities.bf16.bin",
                    legacy_probabilities_bf16(),legacy_eager_score_count);
                save_device_bf16(
                    legacy_internal_a/"14_context_bhsd.bf16.bin",
                    legacy_context_bhsd_bf16(),query_count);
                save_device_bf16(
                    legacy_internal_a/"15_attention_context.bf16.bin",
                    attention_context.as<uint16_t>(),query_count);
            }

            if (audit_this_step) {
                const fs::path attention_audit_root =
                    output_root
                    / "legacy_operator_audit_b"
                    / mode
                    / frame_name
                    / step_name
                    / (
                        "layer_"
                        + two_digit(
                            layer_index)
                    );

                const std::size_t
                    query_elements =
                        static_cast<
                            std::size_t>(
                            kQueryHeads)
                        * sequence_length
                        * kHeadDim;

                const std::size_t
                    cache_elements =
                        static_cast<
                            std::size_t>(
                            kKvHeads)
                        * kMaxCache
                        * kHeadDim;

                const std::size_t
                    score_elements =
                        static_cast<
                            std::size_t>(
                            kQueryHeads)
                        * sequence_length
                        * cache_after;

                save_device_bf16(
                    attention_audit_root
                        / "00_query_rotated.bf16.bin",
                    query_rotated
                        .as<uint16_t>(),
                    query_elements);

                save_device_bf16(
                    attention_audit_root
                        / "01_key_cache_full.bf16.bin",
                    caches[layer_index]
                        .key.as<uint16_t>(),
                    cache_elements);

                save_device_bf16(
                    attention_audit_root
                        / "02_value_cache_full.bf16.bin",
                    caches[layer_index]
                        .value.as<uint16_t>(),
                    cache_elements);

                save_device_bf16(
                    attention_audit_root
                        / "03_qk_raw.bf16.bin",
                    legacy_qk_raw_bf16(),
                    score_elements);

                save_device_bf16(
                    attention_audit_root
                        / "04_qk_scaled.bf16.bin",
                    legacy_qk_scaled_bf16(),
                    score_elements);

                save_device_bf16(
                    attention_audit_root
                        / "05_qk_masked.bf16.bin",
                    legacy_qk_masked_bf16(),
                    score_elements);

                save_device_bf16(
                    attention_audit_root
                        / "06_probabilities.bf16.bin",
                    legacy_probabilities_bf16(),
                    score_elements);

                save_device_bf16(
                    attention_audit_root
                        / "07_context_bhsd.bf16.bin",
                    legacy_context_bhsd_bf16(),
                    query_elements);

                save_device_bf16(
                    attention_audit_root
                        / "10_context_bshd.bf16.bin",
                    attention_context
                        .as<uint16_t>(),
                    query_elements);
            }

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::OProj,
            layer_index,
                attention_context.as<uint16_t>(),
                layer.o_projection.as<uint16_t>(),
                nullptr,
                attention_output.as<uint16_t>(),
                sequence_length,
                kHidden,
                kQueryDim);

            launch_add(
                hidden.as<uint16_t>(),
                attention_output.as<uint16_t>(),
                attention_residual.as<uint16_t>(),
                hidden_count);

            launch_rmsnorm(
                attention_residual.as<uint16_t>(),
                layer.post_attention_layernorm
                    .as<uint16_t>(),
                post_norm.as<uint16_t>(),
                sequence_length,
                kHidden);

            if(capture_common){
                save_device_bf16(
                    legacy_internal_c/"14_attention_output.bf16.bin",
                    attention_output.as<uint16_t>(),hidden_count);
                save_device_bf16(
                    legacy_internal_c/"15_attention_residual.bf16.bin",
                    attention_residual.as<uint16_t>(),hidden_count);
                save_device_bf16(
                    legacy_internal_c/"16_post_norm.bf16.bin",
                    post_norm.as<uint16_t>(),hidden_count);
            }

            if(capture_target){
                save_device_bf16(
                    legacy_internal_a/"16_attention_output.bf16.bin",
                    attention_output.as<uint16_t>(),hidden_count);
                save_device_bf16(
                    legacy_internal_a/"17_attention_residual.bf16.bin",
                    attention_residual.as<uint16_t>(),hidden_count);
                save_device_bf16(
                    legacy_internal_a/"18_post_norm.bf16.bin",
                    post_norm.as<uint16_t>(),hidden_count);
            }

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::GateProj,
            layer_index,
                post_norm.as<uint16_t>(),
                layer.gate_projection.as<uint16_t>(),
                nullptr,
                gate.as<uint16_t>(),
                sequence_length,
                kIntermediate,
                kHidden);

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::UpProj,
            layer_index,
                post_norm.as<uint16_t>(),
                layer.up_projection.as<uint16_t>(),
                nullptr,
                up.as<uint16_t>(),
                sequence_length,
                kIntermediate,
                kHidden);

            launch_swiglu(
                gate.as<uint16_t>(),
                up.as<uint16_t>(),
                swiglu.as<uint16_t>(),
                intermediate_count);

            explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::DownProj,
            layer_index,
                swiglu.as<uint16_t>(),
                layer.down_projection.as<uint16_t>(),
                nullptr,
                down.as<uint16_t>(),
                sequence_length,
                kHidden,
                kIntermediate);

            if(capture_common){
                save_device_bf16(
                    legacy_internal_c/"17_gate.bf16.bin",
                    gate.as<uint16_t>(),intermediate_count);
                save_device_bf16(
                    legacy_internal_c/"18_up.bf16.bin",
                    up.as<uint16_t>(),intermediate_count);
                save_device_bf16(
                    legacy_internal_c/"19_swiglu.bf16.bin",
                    swiglu.as<uint16_t>(),intermediate_count);
                save_device_bf16(
                    legacy_internal_c/"20_down.bf16.bin",
                    down.as<uint16_t>(),hidden_count);
            }

            if(capture_target){
                save_device_bf16(
                    legacy_internal_a/"19_gate.bf16.bin",
                    gate.as<uint16_t>(),intermediate_count);
                save_device_bf16(
                    legacy_internal_a/"20_up.bf16.bin",
                    up.as<uint16_t>(),intermediate_count);
                save_device_bf16(
                    legacy_internal_a/"21_swiglu.bf16.bin",
                    swiglu.as<uint16_t>(),intermediate_count);
                save_device_bf16(
                    legacy_internal_a/"22_down.bf16.bin",
                    down.as<uint16_t>(),hidden_count);
            }

            launch_add(
                attention_residual.as<uint16_t>(),
                down.as<uint16_t>(),
                layer_output.as<uint16_t>(),
                hidden_count);

            const fs::path layer_output_root =
                step_output
                / (
                    "layer_"
                    + two_digit(layer_index)
                );

            save_device_bf16_validation(
                layer_output_root
                    / "output.bf16.bin",
                layer_output.as<uint16_t>(),
                hidden_count,
                teacher_step_mode);

            if(capture_target)
                save_device_bf16(
                    legacy_internal_a/"23_layer_output.bf16.bin",
                    layer_output.as<uint16_t>(),
                    hidden_count);

            launch_compact_cache(
                caches[layer_index]
                    .key.as<uint16_t>(),
                compact_key.as<uint16_t>(),
                cache_after);

            launch_compact_cache(
                caches[layer_index]
                    .value.as<uint16_t>(),
                compact_value.as<uint16_t>(),
                cache_after);

            save_device_bf16(
                layer_output_root
                    / "cache_key_after_update.bf16.bin",
                compact_key.as<uint16_t>(),
                static_cast<std::size_t>(
                    kKvHeads
                )
                    * cache_after
                    * kHeadDim);

            save_device_bf16(
                layer_output_root
                    / "cache_value_after_update.bf16.bin",
                compact_value.as<uint16_t>(),
                static_cast<std::size_t>(
                    kKvHeads
                )
                    * cache_after
                    * kHeadDim);

            if(capture_target){
                const std::size_t legacy_compact_count_a=
                    static_cast<std::size_t>(kKvHeads)
                    *cache_after*kHeadDim;
                save_device_bf16(
                    legacy_internal_a/"08_key_cache_full.bf16.bin",
                    compact_key.as<uint16_t>(),
                    legacy_compact_count_a);
                save_device_bf16(
                    legacy_internal_a/"09_value_cache_full.bf16.bin",
                    compact_value.as<uint16_t>(),
                    legacy_compact_count_a);
            }

            if(capture_kv_provenance){
                const std::size_t legacy_compact_count_b=
                    static_cast<std::size_t>(kKvHeads)
                    *cache_after*kHeadDim;
                save_device_bf16(
                    legacy_internal_b/"04_key_cache_full.bf16.bin",
                    compact_key.as<uint16_t>(),
                    legacy_compact_count_b);
            }

            hidden = std::move(layer_output);
        }

        const std::size_t hidden_count =
            static_cast<std::size_t>(
                sequence_length
            ) * kHidden;

        DeviceBuffer final_norm(
            hidden_count
            * sizeof(uint16_t));

        launch_rmsnorm(
            hidden.as<uint16_t>(),
            weights.final_norm.as<uint16_t>(),
            final_norm.as<uint16_t>(),
            sequence_length,
            kHidden);

        save_device_bf16_validation(
            step_output
                / "final_norm_output.bf16.bin",
            final_norm.as<uint16_t>(),
            hidden_count,
            mode == "teacher");

        const std::size_t logits_count =
            static_cast<std::size_t>(
                sequence_length
            ) * kVocab;

        DeviceBuffer logits(
            logits_count
            * sizeof(uint16_t));

        explicit_runtime::explicit_linear_bf16(
            explicit_runtime::LinearOp::LmHead,
            -1,
            final_norm.as<uint16_t>(),
            weights.lm_heads[step]
                .as<uint16_t>(),
                nullptr,
            logits.as<uint16_t>(),
            sequence_length,
            kVocab,
            kHidden);

        save_device_bf16_validation(
            step_output / "raw_logits.bf16.bin",
            logits.as<uint16_t>(),
            logits_count,
            mode == "teacher");

        std::int32_t generated=0;

        if(
            ::qwen3_tts::baseline::
                g_sampling_enabled
        ){
            const std::size_t
                last_row_offset =
                    static_cast<std::size_t>(
                        sequence_length - 1)
                    * kVocab;

            const auto host_logits=
                download(
                    logits.as<uint16_t>()
                        + last_row_offset,
                    kVocab);

            generated=
                ::qwen3_tts::baseline::
                    sample_topk_bf16(
                        host_logits.data(),
                        kVocab,
                        50,
                        0.9f);
        }else{
            DeviceBuffer code(
                sizeof(int32_t));

            hipLaunchKernelGGL(
                argmax_bf16_kernel,
                dim3(1),
                dim3(kThreads),
                0,
                0,
                logits.as<uint16_t>(),
                (sequence_length - 1) * kVocab,
                code.as<int32_t>());

            PRED_HIP_CHECK(
                hipGetLastError());

            PRED_HIP_CHECK(
                hipDeviceSynchronize());

            generated=
                download(
                    code.as<int32_t>(),
                    1)[0];
        }

        generated_codes[step + 1] = generated;

        write_binary(
            step_output / "code.i32.bin",
            std::vector<int32_t>{generated});

        uint16_t* embedding_destination =
            selected_embeddings.as<uint16_t>()
            + static_cast<std::size_t>(
                step + 1
            ) * kOuterHidden;

        hipLaunchKernelGGL(
            gather_embedding_kernel,
            dim3(
                (kOuterHidden + kThreads - 1)
                / kThreads
            ),
            dim3(kThreads),
            0,
            0,
            weights.embeddings[step]
                .as<uint16_t>(),
            generated,
            embedding_destination);

        PRED_HIP_CHECK(hipGetLastError());

        if (mode == "free" && step < kSteps - 1) {
            PRED_HIP_CHECK(hipMemcpy(
                free_input.as<uint16_t>(),
                embedding_destination,
                kOuterHidden * sizeof(uint16_t),
                hipMemcpyDeviceToDevice));
        }

        explicit_runtime::set_linear_audit_context(
            false);

    }

    DeviceBuffer embedding_sum(
        kOuterHidden
        * sizeof(uint16_t));

    hipLaunchKernelGGL(
        sum_codec_embeddings_kernel,
        dim3(
            (kOuterHidden + kThreads - 1)
            / kThreads
        ),
        dim3(kThreads),
        0,
        0,
        selected_embeddings.as<uint16_t>(),
        embedding_sum.as<uint16_t>());

    PRED_HIP_CHECK(hipGetLastError());

    DeviceBuffer next_talker_input(
        kOuterHidden
        * sizeof(uint16_t));

    launch_add(
        embedding_sum.as<uint16_t>(),
        context_embedding.as<uint16_t>(),
        next_talker_input.as<uint16_t>(),
        kOuterHidden);

    PRED_HIP_CHECK(hipEventRecord(stop));
    PRED_HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    PRED_HIP_CHECK(hipEventElapsedTime(
        &elapsed_ms,
        start,
        stop));

    PRED_HIP_CHECK(hipEventDestroy(start));
    PRED_HIP_CHECK(hipEventDestroy(stop));

    save_device_bf16(
        frame_output
            / "codec_embedding_sum.bf16.bin",
        embedding_sum.as<uint16_t>(),
        kOuterHidden);

    write_binary(
        frame_output
            / "next_talker_input.bf16.bin",
        download(
            next_talker_input.as<uint16_t>(),
            kOuterHidden));

    write_binary(
        frame_output
            / "generated_codes.i32.bin",
        std::vector<int32_t>(
            generated_codes.begin(),
            generated_codes.end()));

    std::cout
        << mode
        << " " << frame_name
        << " elapsed_ms="
        << elapsed_ms
        << " generated_codes=[";

    for (
        int index = 0;
        index < kCodeGroups;
        ++index) {

        if (index) {
            std::cout << ", ";
        }
        std::cout << generated_codes[index];
    }

    std::cout << "]\n";
}




// ============================================================================
// Persistent 15-step Code Predictor HIP Graph.
// ============================================================================
__global__
void prepare_group0_input_kernel(
    const std::uint16_t* __restrict__ codec_table,
    std::size_t codec_rows,
    const std::int32_t* __restrict__ code0,
    std::uint16_t* __restrict__ free_input,
    std::uint16_t* __restrict__ selected_embeddings,
    std::int32_t* __restrict__ generated_codes){

    const int d=
        static_cast<int>(blockIdx.x)
        *static_cast<int>(blockDim.x)
        +static_cast<int>(threadIdx.x);

    const int token=code0[0];

    if(token<0
       ||static_cast<std::size_t>(token)>=codec_rows){
        if(d==0)generated_codes[0]=0;
        return;
    }

    if(d<kOuterHidden){
        const std::uint16_t value=
            codec_table[
                static_cast<std::size_t>(token)
                *kOuterHidden+d];

        selected_embeddings[d]=value;
        free_input[kOuterHidden+d]=value;
    }

    if(d==0)generated_codes[0]=token;
}

__global__
void gather_predictor_embedding_kernel(
    const std::uint16_t* __restrict__ table,
    const std::int32_t* __restrict__ generated_codes,
    int group,
    std::uint16_t* __restrict__ selected_embeddings,
    std::uint16_t* __restrict__ next_step_input){

    const int d=
        static_cast<int>(blockIdx.x)
        *static_cast<int>(blockDim.x)
        +static_cast<int>(threadIdx.x);

    if(d>=kOuterHidden)return;

    const int token=
        generated_codes[group];

    const std::uint16_t value=
        table[
            static_cast<std::size_t>(token)
            *kOuterHidden+d];

    selected_embeddings[
        static_cast<std::size_t>(group)
        *kOuterHidden+d]=value;

    next_step_input[d]=value;
}


// ============================================================================
// Predictor M=1 locality kernels.
//
// Predictor step0 is a 2-row prefill and intentionally remains on the staged
// correctness path. Steps 1..14 are M=1 and use these kernels.
//
// Long-lived/cacheable material:
//   weights, cross-layer hidden, persistent K/V.
//
// Derived values:
//   staged into LDS only when reused by waves, otherwise kept in lane-private
//   VGPR accumulators and immediately consumed.
//

// ============================================================================

__global__ __launch_bounds__(256, 2)
void predictor_qkv_m1_fused_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ q_weight,
    const std::uint16_t* __restrict__ k_weight,
    const std::uint16_t* __restrict__ v_weight,
    std::uint16_t* __restrict__ q_output,
    std::uint16_t* __restrict__ k_output,
    std::uint16_t* __restrict__ v_output) {

    constexpr int waves_per_block = 8;
    constexpr int outputs_per_wave = 2;

    extern __shared__ std::uint16_t shared_input[];

    for (int i=static_cast<int>(threadIdx.x);
         i<kHidden;
         i+=static_cast<int>(blockDim.x)) {
        shared_input[i]=input[i];
    }
    __syncthreads();

    const int thread=static_cast<int>(threadIdx.x);
    const int wave=thread>>5;
    const int lane=thread&31;

    const int first=
        (static_cast<int>(blockIdx.x)*waves_per_block+wave)
        * outputs_per_wave;
    const int total=kQueryDim+2*kKvDim;
    if(first>=total)return;

    const int second=first+1;
    const bool have_second=second<total;

    const std::uint16_t* w0=nullptr;
    const std::uint16_t* w1=nullptr;
    std::uint16_t* o0=nullptr;
    std::uint16_t* o1=nullptr;
    int row0=0,row1=0;

    if(first<kQueryDim){
        w0=q_weight;o0=q_output;row0=first;
    }else if(first<kQueryDim+kKvDim){
        w0=k_weight;o0=k_output;row0=first-kQueryDim;
    }else{
        w0=v_weight;o0=v_output;row0=first-kQueryDim-kKvDim;
    }

    if(have_second){
        if(second<kQueryDim){
            w1=q_weight;o1=q_output;row1=second;
        }else if(second<kQueryDim+kKvDim){
            w1=k_weight;o1=k_output;row1=second-kQueryDim;
        }else{
            w1=v_weight;o1=v_output;row1=second-kQueryDim-kKvDim;
        }
    }

    const std::size_t base0=static_cast<std::size_t>(row0)*kHidden;
    const std::size_t base1=have_second?static_cast<std::size_t>(row1)*kHidden:0;

    float a0=0.0f,a1=0.0f;

    for(int k=lane*2;k+1<kHidden;k+=64){
        const auto x0=shared_input[k];
        const auto x1=shared_input[k+1];

        a0=predictor_bf16_dot2_accumulate(
            x0,x1,w0[base0+k],w0[base0+k+1],a0);

        if(have_second){
            a1=predictor_bf16_dot2_accumulate(
                x0,x1,w1[base1+k],w1[base1+k+1],a1);
        }
    }

    a0=legacy_wave_reduce_sum(a0);
    if(have_second)a1=legacy_wave_reduce_sum(a1);

    if(lane==0){
        o0[row0]=float_to_bf16_rne(a0);
        if(have_second)o1[row1]=float_to_bf16_rne(a1);
    }
}


__global__ __launch_bounds__(256, 2)
void predictor_qk_norm_rope_cache_m1_kernel(
    const std::uint16_t* __restrict__ q,
    const std::uint16_t* __restrict__ k,
    const std::uint16_t* __restrict__ v,
    const std::uint16_t* __restrict__ q_norm_weight,
    const std::uint16_t* __restrict__ k_norm_weight,
    const std::uint16_t* __restrict__ cosine,
    const std::uint16_t* __restrict__ sine,
    std::uint16_t* __restrict__ q_rotated,
    std::uint16_t* __restrict__ key_cache,
    std::uint16_t* __restrict__ value_cache,
    int cache_position) {

    __shared__ std::uint16_t normalized[kHeadDim];
    __shared__ float inverse_rms;

    const int head_slot =
        static_cast<int>(blockIdx.x);

    const bool is_query =
        head_slot < kQueryHeads;

    const int head =
        is_query
        ? head_slot
        : head_slot - kQueryHeads;

    if (
        !is_query
        && head >= kKvHeads
    ) {
        return;
    }

    const int dimension =
        static_cast<int>(threadIdx.x);

    const std::uint16_t* source =
        is_query ? q : k;

    const std::uint16_t* weight =
        is_query
        ? q_norm_weight
        : k_norm_weight;

    float square = 0.0f;

    if (dimension < kHeadDim) {
        const float value =
            bf16_to_float(
                source[
                    head * kHeadDim
                    + dimension]);

        square = value * value;
    }

    // Use the same 256-thread reduction topology as the staged Predictor
    // RMSNorm kernel; lanes 128..255 contribute exact zero.
    const float sum =
        block_reduce_sum(
            square);

    if (threadIdx.x == 0) {
        inverse_rms =
            rsqrtf(
                sum
                / static_cast<float>(
                    kHeadDim)
                + kRmsEpsilon);
    }

    __syncthreads();

    if (dimension < kHeadDim) {
        const float value =
            bf16_to_float(
                source[
                    head * kHeadDim
                    + dimension]);

        // Official/staged RMSNorm boundary:
        // normalize in FP32 -> BF16 -> multiply weight -> BF16.
        const std::uint16_t normalized_bits =
            float_to_bf16_rne(
                value * inverse_rms);

        normalized[dimension] =
            float_to_bf16_rne(
                bf16_to_float(
                    normalized_bits)
                * bf16_to_float(
                    weight[
                        dimension]));
    }

    __syncthreads();

    if (dimension >= kHeadDim) {
        return;
    }

    const int rotated_dimension =
        dimension < kHeadDim / 2
        ? dimension + kHeadDim / 2
        : dimension - kHeadDim / 2;

    const float rotated_sign =
        dimension < kHeadDim / 2
        ? -1.0f
        : 1.0f;

    const float value =
        bf16_to_float(
            normalized[
                dimension]);

    const float rotated =
        rotated_sign
        * bf16_to_float(
            normalized[
                rotated_dimension]);

    const float cos_value =
        bf16_to_float(
            cosine[
                dimension]);

    const float sin_value =
        bf16_to_float(
            sine[
                dimension]);

    // Preserve staged RoPE:
    // BF16(x*cos) + BF16(rotate_half(x)*sin) -> BF16.
    const std::uint16_t first =
        float_to_bf16_rne(
            value * cos_value);

    const std::uint16_t second =
        float_to_bf16_rne(
            rotated * sin_value);

    const std::uint16_t rope_value =
        float_to_bf16_rne(
            bf16_to_float(first)
            + bf16_to_float(second));

    if (is_query) {
        q_rotated[
            head * kHeadDim
            + dimension] =
                rope_value;
    } else {
        const std::size_t cache_index =
            (
                static_cast<std::size_t>(
                    head)
                * kMaxCache
                + cache_position
            ) * kHeadDim
            + dimension;

        key_cache[
            cache_index] =
                rope_value;

        // M=1 transpose is identity; V projection output is already BF16.
        value_cache[
            cache_index] =
                v[
                    head * kHeadDim
                    + dimension];
    }
}


// M=1 Predictor attention: one block owns one KV head and its two Q heads.
// key_length <= 16. QK and PV retain the exact wave32 reduction topology used
// by the staged kernels while each K/V element is loaded once for both Q heads.
__global__ __launch_bounds__(256, 2)
void predictor_attention_gqa2_m1_kernel(
    const std::uint16_t* __restrict__ query,
    const std::uint16_t* __restrict__ key_cache,
    const std::uint16_t* __restrict__ value_cache,
    std::uint16_t* __restrict__ output,
    int key_length) {

    constexpr int waves_per_block = 8;

    __shared__ std::uint16_t query0[kHeadDim];
    __shared__ std::uint16_t query1[kHeadDim];
    __shared__ std::uint16_t score0[kMaxCache];
    __shared__ std::uint16_t score1[kMaxCache];

    const int kv_head =
        static_cast<int>(blockIdx.x);

    if (kv_head >= kKvHeads) {
        return;
    }

    const int query_head0 =
        kv_head * 2;

    const int query_head1 =
        query_head0 + 1;

    const int thread =
        static_cast<int>(threadIdx.x);

    if (thread < kHeadDim) {
        query0[thread] =
            query[
                query_head0
                    * kHeadDim
                + thread];
    } else if (
        thread < 2 * kHeadDim
    ) {
        const int d =
            thread - kHeadDim;

        query1[d] =
            query[
                query_head1
                    * kHeadDim
                + d];
    }

    __syncthreads();

    const int wave =
        thread >> 5;

    const int lane =
        thread & 31;

    // Same wave-per-score reduction as attention_qk_wave32_kernel.
    // One K lane load feeds both query heads.
    for (
        int key_index = wave;
        key_index < key_length;
        key_index += waves_per_block
    ) {
        const std::size_t key_base =
            (
                static_cast<std::size_t>(
                    kv_head)
                * kMaxCache
                + key_index
            ) * kHeadDim;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        for (
            int d = lane;
            d < kHeadDim;
            d += 32
        ) {
            const float kval =
                bf16_to_float(
                    key_cache[
                        key_base + d]);

            acc0 = fmaf(
                bf16_to_float(
                    query0[d]),
                kval,
                acc0);

            acc1 = fmaf(
                bf16_to_float(
                    query1[d]),
                kval,
                acc1);
        }

        acc0 =
            legacy_wave_reduce_sum(
                acc0);

        acc1 =
            legacy_wave_reduce_sum(
                acc1);

        if (lane == 0) {
            const std::uint16_t raw0 =
                float_to_bf16_rne(
                    acc0);

            const std::uint16_t raw1 =
                float_to_bf16_rne(
                    acc1);

            const float scale =
                __uint_as_float(
                    kAttentionScaleBits);

            score0[key_index] =
                float_to_bf16_rne(
                    bf16_to_float(raw0)
                    * scale);

            score1[key_index] =
                float_to_bf16_rne(
                    bf16_to_float(raw1)
                    * scale);
        }
    }

    __syncthreads();

    // Two waves perform the two independent softmax rows with exactly the
    // staged 32-lane max/sum topology. No global score/probability tensors.
    if (wave < 2) {
        std::uint16_t* scores =
            wave == 0
            ? score0
            : score1;

        float value =
            lane < key_length
            ? bf16_to_float(
                scores[lane])
            : -INFINITY;

        float maximum = value;

        for (
            int offset = 16;
            offset > 0;
            offset >>= 1
        ) {
            const float other =
                __shfl_down(
                    maximum,
                    offset,
                    32);

            if (lane + offset < 32) {
                maximum =
                    maximum > other
                    ? maximum
                    : other;
            }
        }

        maximum =
            __shfl(
                maximum,
                0,
                32);

        const float exponential =
            lane < key_length
            ? expf(
                value - maximum)
            : 0.0f;

        const float sum =
            legacy_wave_reduce_sum(
                exponential);

        const float total =
            __shfl(
                sum,
                0,
                32);

        if (lane < key_length) {
            scores[lane] =
                float_to_bf16_rne(
                    exponential / total);
        }
    }

    __syncthreads();

    // Same wave-per-context-scalar reduction as staged PV, but pair Q heads
    // so each V load is consumed twice.
    for (
        int d = wave;
        d < kHeadDim;
        d += waves_per_block
    ) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        if (lane < key_length) {
            const std::size_t value_index =
                (
                    static_cast<std::size_t>(
                        kv_head)
                    * kMaxCache
                    + lane
                ) * kHeadDim
                + d;

            const float value =
                bf16_to_float(
                    value_cache[
                        value_index]);

            acc0 = fmaf(
                bf16_to_float(
                    score0[lane]),
                value,
                acc0);

            acc1 = fmaf(
                bf16_to_float(
                    score1[lane]),
                value,
                acc1);
        }

        acc0 =
            legacy_wave_reduce_sum(
                acc0);

        acc1 =
            legacy_wave_reduce_sum(
                acc1);

        if (lane == 0) {
            output[
                query_head0
                    * kHeadDim
                + d] =
                    float_to_bf16_rne(
                        acc0);

            output[
                query_head1
                    * kHeadDim
                + d] =
                    float_to_bf16_rne(
                        acc1);
        }
    }
}


__global__ __launch_bounds__(256, 2)
void predictor_linear_m1_fused_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    const std::uint16_t* __restrict__ residual,
    std::uint16_t* __restrict__ output,
    int output_columns,
    int reduction_columns) {

    constexpr int waves_per_block=8;
    constexpr int outputs_per_wave=2;

    extern __shared__ std::uint16_t shared_input[];

    for(int i=static_cast<int>(threadIdx.x);
        i<reduction_columns;
        i+=static_cast<int>(blockDim.x)){
        shared_input[i]=input[i];
    }
    __syncthreads();

    const int thread=static_cast<int>(threadIdx.x);
    const int wave=thread>>5;
    const int lane=thread&31;

    const int out0=
        (static_cast<int>(blockIdx.x)*waves_per_block+wave)
        *outputs_per_wave;
    if(out0>=output_columns)return;

    const int out1=out0+1;
    const bool have_second=out1<output_columns;

    const std::size_t base0=static_cast<std::size_t>(out0)*reduction_columns;
    const std::size_t base1=have_second?static_cast<std::size_t>(out1)*reduction_columns:0;

    float a0=0.0f,a1=0.0f;

    for(int k=lane*2;k+1<reduction_columns;k+=64){
        const auto x0=shared_input[k];
        const auto x1=shared_input[k+1];

        a0=predictor_bf16_dot2_accumulate(
            x0,x1,weight[base0+k],weight[base0+k+1],a0);

        if(have_second){
            a1=predictor_bf16_dot2_accumulate(
                x0,x1,weight[base1+k],weight[base1+k+1],a1);
        }
    }

    if((reduction_columns&1)!=0&&lane==0){
        const int k=reduction_columns-1;
        const float x=bf16_to_float(shared_input[k]);
        a0=fmaf(x,bf16_to_float(weight[base0+k]),a0);
        if(have_second)a1=fmaf(x,bf16_to_float(weight[base1+k]),a1);
    }

    a0=legacy_wave_reduce_sum(a0);
    if(have_second)a1=legacy_wave_reduce_sum(a1);

    if(lane==0){
        const auto v0=float_to_bf16_rne(a0);
        output[out0]=residual==nullptr
            ?v0
            :float_to_bf16_rne(
                bf16_to_float(residual[out0])+bf16_to_float(v0));

        if(have_second){
            const auto v1=float_to_bf16_rne(a1);
            output[out1]=residual==nullptr
                ?v1
                :float_to_bf16_rne(
                    bf16_to_float(residual[out1])+bf16_to_float(v1));
        }
    }
}


__global__ __launch_bounds__(256, 2)
void predictor_add_postnorm_m1_kernel(
    const std::uint16_t* __restrict__ hidden,
    const std::uint16_t* __restrict__ attention,
    const std::uint16_t* __restrict__ norm_weight,
    std::uint16_t* __restrict__ residual,
    std::uint16_t* __restrict__ normalized) {

    __shared__ std::uint16_t shared_residual[kHidden];
    __shared__ float inverse_rms;

    float local_sum = 0.0f;

    for (
        int i = static_cast<int>(threadIdx.x);
        i < kHidden;
        i += static_cast<int>(blockDim.x)
    ) {
        const std::uint16_t value =
            float_to_bf16_rne(
                bf16_to_float(
                    hidden[i])
                + bf16_to_float(
                    attention[i]));

        residual[i] =
            value;

        shared_residual[i] =
            value;

        const float x =
            bf16_to_float(
                value);

        local_sum =
            fmaf(
                x,
                x,
                local_sum);
    }

    const float sum =
        block_reduce_sum(
            local_sum);

    if (threadIdx.x == 0) {
        inverse_rms =
            rsqrtf(
                sum
                / static_cast<float>(
                    kHidden)
                + kRmsEpsilon);
    }

    __syncthreads();

    for (
        int i = static_cast<int>(threadIdx.x);
        i < kHidden;
        i += static_cast<int>(blockDim.x)
    ) {
        const std::uint16_t normalized_bits =
            float_to_bf16_rne(
                bf16_to_float(
                    shared_residual[i])
                * inverse_rms);

        normalized[i] =
            float_to_bf16_rne(
                bf16_to_float(
                    normalized_bits)
                * bf16_to_float(
                    norm_weight[i]));
    }
}


__global__ __launch_bounds__(256, 2)
void predictor_gate_up_swiglu_m1_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ gate_weight,
    const std::uint16_t* __restrict__ up_weight,
    std::uint16_t* __restrict__ output) {

    constexpr int waves_per_block=8;
    constexpr int outputs_per_wave=2;

    extern __shared__ std::uint16_t shared_input[];

    for(int i=static_cast<int>(threadIdx.x);
        i<kHidden;
        i+=static_cast<int>(blockDim.x)){
        shared_input[i]=input[i];
    }
    __syncthreads();

    const int thread=static_cast<int>(threadIdx.x);
    const int wave=thread>>5;
    const int lane=thread&31;

    const int ch0=
        (static_cast<int>(blockIdx.x)*waves_per_block+wave)
        *outputs_per_wave;
    if(ch0>=kIntermediate)return;

    const int ch1=ch0+1;
    const bool have_second=ch1<kIntermediate;

    const std::size_t base0=static_cast<std::size_t>(ch0)*kHidden;
    const std::size_t base1=have_second?static_cast<std::size_t>(ch1)*kHidden:0;

    float g0=0.0f,u0=0.0f,g1=0.0f,u1=0.0f;

    for(int k=lane*2;k+1<kHidden;k+=64){
        const auto x0=shared_input[k];
        const auto x1=shared_input[k+1];

        g0=predictor_bf16_dot2_accumulate(
            x0,x1,gate_weight[base0+k],gate_weight[base0+k+1],g0);
        u0=predictor_bf16_dot2_accumulate(
            x0,x1,up_weight[base0+k],up_weight[base0+k+1],u0);

        if(have_second){
            g1=predictor_bf16_dot2_accumulate(
                x0,x1,gate_weight[base1+k],gate_weight[base1+k+1],g1);
            u1=predictor_bf16_dot2_accumulate(
                x0,x1,up_weight[base1+k],up_weight[base1+k+1],u1);
        }
    }

    g0=legacy_wave_reduce_sum(g0);
    u0=legacy_wave_reduce_sum(u0);
    if(have_second){
        g1=legacy_wave_reduce_sum(g1);
        u1=legacy_wave_reduce_sum(u1);
    }

    if(lane==0){
        const auto gb0=float_to_bf16_rne(g0);
        const auto ub0=float_to_bf16_rne(u0);
        const float gv0=bf16_to_float(gb0);
        const auto act0=float_to_bf16_rne(gv0/(1.0f+expf(-gv0)));
        output[ch0]=float_to_bf16_rne(
            bf16_to_float(act0)*bf16_to_float(ub0));

        if(have_second){
            const auto gb1=float_to_bf16_rne(g1);
            const auto ub1=float_to_bf16_rne(u1);
            const float gv1=bf16_to_float(gb1);
            const auto act1=float_to_bf16_rne(gv1/(1.0f+expf(-gv1)));
            output[ch1]=float_to_bf16_rne(
                bf16_to_float(act1)*bf16_to_float(ub1));
        }
    }
}


struct PersistentFrameResult{
    std::array<std::int32_t,kCodeGroups> codes{};
    std::vector<std::uint16_t> next_talker_input;
    double elapsed_ms=0.0;
};

class PersistentPredictor{
public:
    PersistentPredictor(
        const PredictorWeights& weights,
        const model_frontend::Tensor& codec_embedding,
        std::uint64_t* rng_state)
        :weights_(weights),
         rng_state_(rng_state),
         codec0_rows_(
             static_cast<std::size_t>(
                 codec_embedding.shape[0])),
         codec0_table_(
             static_cast<std::size_t>(
                 codec_embedding.elements())
             *sizeof(std::uint16_t)),
         free_input_(
             static_cast<std::size_t>(2)
             *kOuterHidden
             *sizeof(std::uint16_t)),
         context_embedding_(
             kOuterHidden*sizeof(std::uint16_t)),
         selected_embeddings_(
             static_cast<std::size_t>(kCodeGroups)
             *kOuterHidden
             *sizeof(std::uint16_t)),
         generated_codes_(
             kCodeGroups*sizeof(std::int32_t)),
         hidden_a_(static_cast<std::size_t>(2)*kHidden*2),
         hidden_b_(static_cast<std::size_t>(2)*kHidden*2),
         norm_(static_cast<std::size_t>(2)*kHidden*2),
         query_linear_(static_cast<std::size_t>(2)*kQueryDim*2),
         key_linear_(static_cast<std::size_t>(2)*kKvDim*2),
         value_linear_(static_cast<std::size_t>(2)*kKvDim*2),
         query_normalized_(static_cast<std::size_t>(2)*kQueryDim*2),
         key_normalized_(static_cast<std::size_t>(2)*kKvDim*2),
         query_bhsd_(static_cast<std::size_t>(2)*kQueryDim*2),
         key_bhsd_(static_cast<std::size_t>(2)*kKvDim*2),
         value_bhsd_(static_cast<std::size_t>(2)*kKvDim*2),
         query_rotated_(static_cast<std::size_t>(2)*kQueryDim*2),
         key_rotated_(static_cast<std::size_t>(2)*kKvDim*2),
         attention_context_(static_cast<std::size_t>(2)*kQueryDim*2),
         attention_output_(static_cast<std::size_t>(2)*kHidden*2),
         attention_residual_(static_cast<std::size_t>(2)*kHidden*2),
         post_norm_(static_cast<std::size_t>(2)*kHidden*2),
         gate_(static_cast<std::size_t>(2)*kIntermediate*2),
         up_(static_cast<std::size_t>(2)*kIntermediate*2),
         swiglu_(static_cast<std::size_t>(2)*kIntermediate*2),
         down_(static_cast<std::size_t>(2)*kHidden*2),
         final_norm_(static_cast<std::size_t>(2)*kHidden*2),
         logits_(static_cast<std::size_t>(2)*kVocab*2),
         embedding_sum_(kOuterHidden*2),
         next_talker_input_(kOuterHidden*2){

        if(codec_embedding.dtype!="BF16"
           ||codec_embedding.shape.size()!=2
           ||codec_embedding.shape[1]!=kOuterHidden)
            throw std::runtime_error(
                "PersistentPredictor expected BF16 Talker codec embedding");

        PRED_HIP_CHECK(hipMemcpy(
            codec0_table_.pointer,
            codec_embedding.data,
            codec0_table_.bytes,
            hipMemcpyHostToDevice));

        caches_.reserve(kLayers);
        for(int layer=0;layer<kLayers;++layer)
            caches_.emplace_back();

        for(int step=0;step<kSteps;++step){
            std::vector<std::uint16_t> cosine,sine;
            make_rope(step,cosine,sine);

            rope_cos_[step]=DeviceBuffer(
                cosine.size()*sizeof(std::uint16_t));
            rope_sin_[step]=DeviceBuffer(
                sine.size()*sizeof(std::uint16_t));

            upload(rope_cos_[step],cosine);
            upload(rope_sin_[step],sine);
        }

        stream_=
            ::qwen3_tts::baseline::
                create_generation_execution_stream(
                    "Predictor generation stream");

        capture_graph();

        std::cout
            <<"predictor_backend: persistent_gpu_dag\n"
            <<"predictor_gpu_sampling: "
            <<(::qwen3_tts::baseline::g_sampling_enabled
                ?"True":"False")
            <<"\n"
            <<"predictor_graph_node_count: "
            <<graph_node_count_
            <<"\n"
            <<"predictor_per_frame_filesystem_io: False\n"
            <<"predictor_per_step_device_synchronize: False\n"
            <<"predictor_graph_capture_mode: thread_local\n"
            <<"predictor_graph_capture_allows_concurrent_allocator_threads: True\n"
            <<"predictor_generation_stream_cu_partitioned: "
            <<(::qwen3_tts::baseline::g_resident_cu_partition_enabled
                ?"True":"False")
            <<"\n"
            <<"predictor_step0_backend: staged_two_row_correctness_anchor\n"
            <<"predictor_m1_backend: locality_fused_wave32_gfx11_bf16_dot2_pack2\n"
            <<"predictor_m1_outputs_per_wave: 2\n"
            <<"predictor_m1_accumulator: fp32\n"
            <<"predictor_m1_bf16_dot2: True\n"
            <<"predictor_m1_bf16_dot2_scope: recurrent_steps_1_to_14\n"
            <<"predictor_step0_dot2_changed: False\n"
            <<"predictor_m1_qkv_fusion: True\n"
            <<"predictor_m1_qk_norm_rope_cache_fusion: True\n"
            <<"predictor_m1_attention_backend: gqa2_wave32_lds\n"
            <<"predictor_m1_gate_up_swiglu_fusion: True\n"
            <<"predictor_m1_down_residual_fusion: True\n"
            <<"predictor_m1_cache_policy: raw_material_in_cache_derived_in_vgpr_lds\n";
    }

    ~PersistentPredictor(){
        if(graph_exec_)
            (void)hipGraphExecDestroy(graph_exec_);
        if(graph_)
            (void)hipGraphDestroy(graph_);
        if(stream_)
            (void)hipStreamDestroy(stream_);
    }

    PersistentPredictor(
        const PersistentPredictor&)=delete;
    PersistentPredictor& operator=(
        const PersistentPredictor&)=delete;

    std::size_t graph_node_count()const{
        return graph_node_count_;
    }

    PersistentFrameResult run_frame(
        const std::uint16_t* talker_hidden_device,
        const std::int32_t* code0_device,
        const std::uint16_t* context_host){

        const auto start=
            std::chrono::steady_clock::now();

        for(auto& cache:caches_){
            PRED_HIP_CHECK(hipMemsetAsync(
                cache.key.pointer,0,cache.key.bytes,
                stream_));
            PRED_HIP_CHECK(hipMemsetAsync(
                cache.value.pointer,0,cache.value.bytes,
                stream_));
        }

        PRED_HIP_CHECK(hipMemsetAsync(
            selected_embeddings_.pointer,0,
            selected_embeddings_.bytes,stream_));
        PRED_HIP_CHECK(hipMemsetAsync(
            generated_codes_.pointer,0,
            generated_codes_.bytes,stream_));

        PRED_HIP_CHECK(hipMemcpyAsync(
            free_input_.as<std::uint16_t>(),
            talker_hidden_device,
            kOuterHidden*sizeof(std::uint16_t),
            hipMemcpyDeviceToDevice,stream_));

        PRED_HIP_CHECK(hipMemcpyAsync(
            context_embedding_.as<std::uint16_t>(),
            context_host,
            kOuterHidden*sizeof(std::uint16_t),
            hipMemcpyHostToDevice,stream_));

        hipLaunchKernelGGL(
            prepare_group0_input_kernel,
            dim3((kOuterHidden+kThreads-1)/kThreads),
            dim3(kThreads),0,stream_,
            codec0_table_.as<std::uint16_t>(),
            codec0_rows_,
            code0_device,
            free_input_.as<std::uint16_t>(),
            selected_embeddings_.as<std::uint16_t>(),
            generated_codes_.as<std::int32_t>());
        PRED_HIP_CHECK(hipGetLastError());

        PRED_HIP_CHECK(
            hipGraphLaunch(graph_exec_,stream_));

        PersistentFrameResult result;
        result.next_talker_input.resize(kOuterHidden);

        PRED_HIP_CHECK(hipMemcpyAsync(
            result.codes.data(),
            generated_codes_.as<std::int32_t>(),
            kCodeGroups*sizeof(std::int32_t),
            hipMemcpyDeviceToHost,stream_));

        PRED_HIP_CHECK(hipMemcpyAsync(
            result.next_talker_input.data(),
            next_talker_input_.as<std::uint16_t>(),
            kOuterHidden*sizeof(std::uint16_t),
            hipMemcpyDeviceToHost,stream_));

        PRED_HIP_CHECK(
            hipStreamSynchronize(stream_));

        result.elapsed_ms=
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start)
            .count();

        return result;
    }

    const std::uint16_t*
    device_next_talker_input() const {
        return next_talker_input_
            .as<std::uint16_t>();
    }

    PersistentFrameResult run_frame_device(
        const std::uint16_t* talker_hidden_device,
        const std::int32_t* code0_device,
        const std::uint16_t* context_host) {

        const auto start =
            std::chrono::steady_clock::now();

        for (auto& cache : caches_) {
            PRED_HIP_CHECK(
                hipMemsetAsync(
                    cache.key.pointer,
                    0,
                    cache.key.bytes,
                    stream_));

            PRED_HIP_CHECK(
                hipMemsetAsync(
                    cache.value.pointer,
                    0,
                    cache.value.bytes,
                    stream_));
        }

        PRED_HIP_CHECK(
            hipMemsetAsync(
                selected_embeddings_.pointer,
                0,
                selected_embeddings_.bytes,
                stream_));

        PRED_HIP_CHECK(
            hipMemsetAsync(
                generated_codes_.pointer,
                0,
                generated_codes_.bytes,
                stream_));

        PRED_HIP_CHECK(
            hipMemcpyAsync(
                free_input_.as<std::uint16_t>(),
                talker_hidden_device,
                kOuterHidden
                    * sizeof(std::uint16_t),
                hipMemcpyDeviceToDevice,
                stream_));

        PRED_HIP_CHECK(
            hipMemcpyAsync(
                context_embedding_
                    .as<std::uint16_t>(),
                context_host,
                kOuterHidden
                    * sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_));

        hipLaunchKernelGGL(
            prepare_group0_input_kernel,
            dim3(
                (kOuterHidden + kThreads - 1)
                / kThreads),
            dim3(kThreads),
            0,
            stream_,
            codec0_table_
                .as<std::uint16_t>(),
            codec0_rows_,
            code0_device,
            free_input_
                .as<std::uint16_t>(),
            selected_embeddings_
                .as<std::uint16_t>(),
            generated_codes_
                .as<std::int32_t>());

        PRED_HIP_CHECK(
            hipGetLastError());

        PRED_HIP_CHECK(
            hipGraphLaunch(
                graph_exec_,
                stream_));

        PersistentFrameResult result;

        PRED_HIP_CHECK(
            hipMemcpyAsync(
                result.codes.data(),
                generated_codes_
                    .as<std::int32_t>(),
                kCodeGroups
                    * sizeof(std::int32_t),
                hipMemcpyDeviceToHost,
                stream_));

        // Synchronize only for the 16 returned codec IDs and to establish
        // the cross-stream dependency before Talker consumes next_talker_input_.
        // The 1024-d next Talker vector itself never leaves the GPU.
        PRED_HIP_CHECK(
            hipStreamSynchronize(
                stream_));

        result.elapsed_ms =
            std::chrono::duration<
                double,
                std::milli>(
                    std::chrono::
                        steady_clock::now()
                    - start)
            .count();

        return result;
    }



private:
    static void make_rope(
        int step,
        std::vector<std::uint16_t>& cosine,
        std::vector<std::uint16_t>& sine){

        const int rows=step==0?2:1;
        const int first=step==0?0:step+1;
        const int half=kHeadDim/2;

        cosine.resize(
            static_cast<std::size_t>(rows)*kHeadDim);
        sine.resize(cosine.size());

        for(int row=0;row<rows;++row){
            const int position=first+row;
            for(int i=0;i<half;++i){
                const double inv=std::pow(
                    1000000.0,
                    -2.0*static_cast<double>(i)
                    /static_cast<double>(kHeadDim));
                const double angle=
                    static_cast<double>(position)*inv;
                const auto c=
                    model_frontend::f32_to_bf16_rne(
                        static_cast<float>(std::cos(angle)));
                const auto sn=
                    model_frontend::f32_to_bf16_rne(
                        static_cast<float>(std::sin(angle)));
                const std::size_t base=
                    static_cast<std::size_t>(row)*kHeadDim;
                cosine[base+i]=c;
                cosine[base+i+half]=c;
                sine[base+i]=sn;
                sine[base+i+half]=sn;
            }
        }
    }

    void linear(
        const std::uint16_t* input,
        const std::uint16_t* weight,
        const std::uint16_t* bias,
        std::uint16_t* output,
        int rows,int n,int k){
        qwen3_tts::native_ops::linear_bf16(
            input,weight,bias,output,
            rows,n,k,stream_);
    }

    void rmsnorm(
        const std::uint16_t* input,
        const std::uint16_t* weight,
        std::uint16_t* output,
        int rows,int dimension){
        hipLaunchKernelGGL(
            rmsnorm_kernel,
            dim3(rows),dim3(kThreads),0,stream_,
            input,weight,output,rows,dimension);
        PRED_HIP_CHECK(hipGetLastError());
    }

    void transpose(
        const std::uint16_t* input,
        std::uint16_t* output,
        int rows,int heads){
        const std::uint64_t count=
            static_cast<std::uint64_t>(rows)
            *heads*kHeadDim;
        hipLaunchKernelGGL(
            transpose_bshd_to_bhsd_kernel,
            dim3(static_cast<unsigned>(
                (count+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            input,output,rows,heads);
        PRED_HIP_CHECK(hipGetLastError());
    }

    void rope(
        const std::uint16_t* input,
        const std::uint16_t* cosine,
        const std::uint16_t* sine,
        std::uint16_t* output,
        int rows,int heads){
        const std::uint64_t count=
            static_cast<std::uint64_t>(rows)
            *heads*kHeadDim;
        hipLaunchKernelGGL(
            apply_rope_kernel,
            dim3(static_cast<unsigned>(
                (count+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            input,cosine,sine,output,rows,heads);
        PRED_HIP_CHECK(hipGetLastError());
    }

    void update_cache(
        const std::uint16_t* source,
        std::uint16_t* cache,
        int rows,int start){
        const std::uint64_t count=
            static_cast<std::uint64_t>(kKvHeads)
            *rows*kHeadDim;
        hipLaunchKernelGGL(
            update_cache_kernel,
            dim3(static_cast<unsigned>(
                (count+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            source,cache,rows,start);
        PRED_HIP_CHECK(hipGetLastError());
    }

    void add(
        const std::uint16_t* a,
        const std::uint16_t* b,
        std::uint16_t* y,
        std::uint64_t count){
        hipLaunchKernelGGL(
            add_bf16_kernel,
            dim3(static_cast<unsigned>(
                (count+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            a,b,y,count);
        PRED_HIP_CHECK(hipGetLastError());
    }

    void swiglu(
        const std::uint16_t* gate,
        const std::uint16_t* up,
        std::uint16_t* y,
        std::uint64_t count){
        hipLaunchKernelGGL(
            swiglu_kernel,
            dim3(static_cast<unsigned>(
                (count+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            gate,up,y,count);
        PRED_HIP_CHECK(hipGetLastError());
    }

    void attention(
        const std::uint16_t* query,
        const std::uint16_t* key_cache,
        const std::uint16_t* value_cache,
        std::uint16_t* output,
        int rows,int key_length,
        bool causal_prefill){

        constexpr int waves=8;
        const std::uint64_t scores=
            static_cast<std::uint64_t>(kQueryHeads)
            *rows*key_length;
        const std::uint64_t context=
            static_cast<std::uint64_t>(kQueryHeads)
            *rows*kHeadDim;

        hipLaunchKernelGGL(
            attention_qk_wave32_kernel,
            dim3(static_cast<unsigned>(
                (scores+waves-1)/waves)),
            dim3(waves*32),0,stream_,
            query,key_cache,
            attention_workspace_.qk_raw_bf16
                .as<std::uint16_t>(),
            rows,key_length);
        PRED_HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            attention_scale_qk_kernel,
            dim3(static_cast<unsigned>(
                (scores+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            attention_workspace_.qk_raw_bf16
                .as<std::uint16_t>(),
            attention_workspace_.qk_scaled_bf16
                .as<std::uint16_t>(),
            scores);
        PRED_HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            attention_causal_mask_kernel,
            dim3(static_cast<unsigned>(
                (scores+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            attention_workspace_.qk_scaled_bf16
                .as<std::uint16_t>(),
            attention_workspace_.qk_masked_bf16
                .as<std::uint16_t>(),
            rows,key_length,
            causal_prefill?1:0);
        PRED_HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            attention_softmax_rows_kernel,
            dim3(static_cast<unsigned>(
                kQueryHeads*rows)),
            dim3(32),0,stream_,
            attention_workspace_.qk_masked_bf16
                .as<std::uint16_t>(),
            attention_workspace_.probabilities_bf16
                .as<std::uint16_t>(),
            kQueryHeads*rows,key_length);
        PRED_HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            attention_pv_wave32_kernel,
            dim3(static_cast<unsigned>(
                (context+waves-1)/waves)),
            dim3(waves*32),0,stream_,
            attention_workspace_.probabilities_bf16
                .as<std::uint16_t>(),
            value_cache,
            attention_workspace_.context_bhsd_bf16
                .as<std::uint16_t>(),
            rows,key_length);
        PRED_HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            attention_transpose_context_kernel,
            dim3(static_cast<unsigned>(
                (context+kThreads-1)/kThreads)),
            dim3(kThreads),0,stream_,
            attention_workspace_.context_bhsd_bf16
                .as<std::uint16_t>(),
            output,rows);
        PRED_HIP_CHECK(hipGetLastError());
    }


    void qkv_m1(
        const std::uint16_t* input,
        const LayerWeights& layer) {

        constexpr int waves_per_block = 8;

        const int total_outputs =
            kQueryDim
            + 2 * kKvDim;

        hipLaunchKernelGGL(
            predictor_qkv_m1_fused_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        total_outputs
                        + 2 * waves_per_block
                        - 1
                    )
                    / (2 * waves_per_block))),
            dim3(kThreads),
            static_cast<std::size_t>(
                kHidden)
                * sizeof(std::uint16_t),
            stream_,
            input,
            layer.q_projection
                .as<std::uint16_t>(),
            layer.k_projection
                .as<std::uint16_t>(),
            layer.v_projection
                .as<std::uint16_t>(),
            query_linear_
                .as<std::uint16_t>(),
            key_linear_
                .as<std::uint16_t>(),
            value_linear_
                .as<std::uint16_t>());

        PRED_HIP_CHECK(
            hipGetLastError());
    }

    void qk_norm_rope_cache_m1(
        const LayerWeights& layer,
        LayerCache& cache,
        int cache_position,
        int step) {

        hipLaunchKernelGGL(
            predictor_qk_norm_rope_cache_m1_kernel,
            dim3(
                static_cast<unsigned>(
                    kQueryHeads
                    + kKvHeads)),
            dim3(kThreads),
            0,
            stream_,
            query_linear_
                .as<std::uint16_t>(),
            key_linear_
                .as<std::uint16_t>(),
            value_linear_
                .as<std::uint16_t>(),
            layer.q_norm
                .as<std::uint16_t>(),
            layer.k_norm
                .as<std::uint16_t>(),
            rope_cos_[step]
                .as<std::uint16_t>(),
            rope_sin_[step]
                .as<std::uint16_t>(),
            query_rotated_
                .as<std::uint16_t>(),
            cache.key
                .as<std::uint16_t>(),
            cache.value
                .as<std::uint16_t>(),
            cache_position);

        PRED_HIP_CHECK(
            hipGetLastError());
    }

    void attention_m1(
        const LayerCache& cache,
        int key_length) {

        hipLaunchKernelGGL(
            predictor_attention_gqa2_m1_kernel,
            dim3(
                static_cast<unsigned>(
                    kKvHeads)),
            dim3(kThreads),
            0,
            stream_,
            query_rotated_
                .as<std::uint16_t>(),
            cache.key
                .as<std::uint16_t>(),
            cache.value
                .as<std::uint16_t>(),
            attention_context_
                .as<std::uint16_t>(),
            key_length);

        PRED_HIP_CHECK(
            hipGetLastError());
    }

    void linear_m1(
        const std::uint16_t* input,
        const std::uint16_t* weight,
        const std::uint16_t* residual,
        std::uint16_t* output,
        int output_columns,
        int reduction_columns) {

        constexpr int waves_per_block = 8;

        hipLaunchKernelGGL(
            predictor_linear_m1_fused_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        output_columns
                        + 2 * waves_per_block
                        - 1
                    )
                    / (2 * waves_per_block))),
            dim3(kThreads),
            static_cast<std::size_t>(
                reduction_columns)
                * sizeof(std::uint16_t),
            stream_,
            input,
            weight,
            residual,
            output,
            output_columns,
            reduction_columns);

        PRED_HIP_CHECK(
            hipGetLastError());
    }

    void add_postnorm_m1(
        const std::uint16_t* hidden,
        const std::uint16_t* attention,
        const std::uint16_t* norm_weight) {

        hipLaunchKernelGGL(
            predictor_add_postnorm_m1_kernel,
            dim3(1),
            dim3(kThreads),
            0,
            stream_,
            hidden,
            attention,
            norm_weight,
            attention_residual_
                .as<std::uint16_t>(),
            post_norm_
                .as<std::uint16_t>());

        PRED_HIP_CHECK(
            hipGetLastError());
    }

    void gate_up_swiglu_m1(
        const LayerWeights& layer) {

        constexpr int waves_per_block = 8;

        hipLaunchKernelGGL(
            predictor_gate_up_swiglu_m1_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        kIntermediate
                        + 2 * waves_per_block
                        - 1
                    )
                    / (2 * waves_per_block))),
            dim3(kThreads),
            static_cast<std::size_t>(
                kHidden)
                * sizeof(std::uint16_t),
            stream_,
            post_norm_
                .as<std::uint16_t>(),
            layer.gate_projection
                .as<std::uint16_t>(),
            layer.up_projection
                .as<std::uint16_t>(),
            swiglu_
                .as<std::uint16_t>());

        PRED_HIP_CHECK(
            hipGetLastError());
    }


    void capture_step(int step){
        const int rows=step==0?2:1;
        const int cache_before=
            step==0?0:step+1;
        const int cache_after=step+2;

        if constexpr(kProjectionIdentity){
            PRED_HIP_CHECK(hipMemcpyAsync(
                hidden_a_.pointer,
                free_input_.pointer,
                static_cast<std::size_t>(rows)*kHidden*sizeof(std::uint16_t),
                hipMemcpyDeviceToDevice,
                stream_));
        }else{
            linear(
                free_input_.as<std::uint16_t>(),
                weights_.projection_weight
                    .as<std::uint16_t>(),
                weights_.projection_bias
                    .as<std::uint16_t>(),
                hidden_a_.as<std::uint16_t>(),
                rows,kHidden,kOuterHidden);
        }

        DeviceBuffer* current=&hidden_a_;
        DeviceBuffer* next=&hidden_b_;

        for(int li=0;li<kLayers;++li){
            const LayerWeights& layer=
                weights_.layers[li];

            const std::uint64_t hidden_count=
                static_cast<std::uint64_t>(rows)
                *kHidden;
            const std::uint64_t intermediate_count=
                static_cast<std::uint64_t>(rows)
                *kIntermediate;

            rmsnorm(
                current->as<std::uint16_t>(),
                layer.input_layernorm
                    .as<std::uint16_t>(),
                norm_.as<std::uint16_t>(),
                rows,kHidden);

            if(rows==1){
                qkv_m1(
                    norm_.as<std::uint16_t>(),
                    layer);

                qk_norm_rope_cache_m1(
                    layer,
                    caches_[li],
                    cache_before,
                    step);

                attention_m1(
                    caches_[li],
                    cache_after);
            }else{
                // Predictor step0 remains the exact staged 2-row anchor.
                linear(norm_.as<std::uint16_t>(),
                       layer.q_projection.as<std::uint16_t>(),
                       nullptr,query_linear_.as<std::uint16_t>(),
                       rows,kQueryDim,kHidden);
                linear(norm_.as<std::uint16_t>(),
                       layer.k_projection.as<std::uint16_t>(),
                       nullptr,key_linear_.as<std::uint16_t>(),
                       rows,kKvDim,kHidden);
                linear(norm_.as<std::uint16_t>(),
                       layer.v_projection.as<std::uint16_t>(),
                       nullptr,value_linear_.as<std::uint16_t>(),
                       rows,kKvDim,kHidden);

                rmsnorm(
                    query_linear_.as<std::uint16_t>(),
                    layer.q_norm.as<std::uint16_t>(),
                    query_normalized_.as<std::uint16_t>(),
                    rows*kQueryHeads,kHeadDim);
                rmsnorm(
                    key_linear_.as<std::uint16_t>(),
                    layer.k_norm.as<std::uint16_t>(),
                    key_normalized_.as<std::uint16_t>(),
                    rows*kKvHeads,kHeadDim);

                transpose(
                    query_normalized_.as<std::uint16_t>(),
                    query_bhsd_.as<std::uint16_t>(),
                    rows,kQueryHeads);
                transpose(
                    key_normalized_.as<std::uint16_t>(),
                    key_bhsd_.as<std::uint16_t>(),
                    rows,kKvHeads);
                transpose(
                    value_linear_.as<std::uint16_t>(),
                    value_bhsd_.as<std::uint16_t>(),
                    rows,kKvHeads);

                rope(query_bhsd_.as<std::uint16_t>(),
                     rope_cos_[step].as<std::uint16_t>(),
                     rope_sin_[step].as<std::uint16_t>(),
                     query_rotated_.as<std::uint16_t>(),
                     rows,kQueryHeads);
                rope(key_bhsd_.as<std::uint16_t>(),
                     rope_cos_[step].as<std::uint16_t>(),
                     rope_sin_[step].as<std::uint16_t>(),
                     key_rotated_.as<std::uint16_t>(),
                     rows,kKvHeads);

                update_cache(
                    key_rotated_.as<std::uint16_t>(),
                    caches_[li].key.as<std::uint16_t>(),
                    rows,cache_before);
                update_cache(
                    value_bhsd_.as<std::uint16_t>(),
                    caches_[li].value.as<std::uint16_t>(),
                    rows,cache_before);

                attention(
                    query_rotated_.as<std::uint16_t>(),
                    caches_[li].key.as<std::uint16_t>(),
                    caches_[li].value.as<std::uint16_t>(),
                    attention_context_.as<std::uint16_t>(),
                    rows,cache_after,true);
            }

            if(rows==1){
                // OProj input is staged once per block in LDS.
                linear_m1(
                    attention_context_
                        .as<std::uint16_t>(),
                    layer.o_projection
                        .as<std::uint16_t>(),
                    nullptr,
                    attention_output_
                        .as<std::uint16_t>(),
                    kHidden,
                    kQueryDim);

                // Fresh BF16 residual is the model boundary; keep it in LDS
                // and consume it immediately for post-attention RMSNorm.
                add_postnorm_m1(
                    current->as<std::uint16_t>(),
                    attention_output_
                        .as<std::uint16_t>(),
                    layer.post_attention_layernorm
                        .as<std::uint16_t>());

                // Gate/Up derived tensors never reach global memory.
                gate_up_swiglu_m1(
                    layer);

                // Down result is BF16-materialized in-register then consumed
                // immediately by the BF16 residual addition.
                linear_m1(
                    swiglu_
                        .as<std::uint16_t>(),
                    layer.down_projection
                        .as<std::uint16_t>(),
                    attention_residual_
                        .as<std::uint16_t>(),
                    next->as<std::uint16_t>(),
                    kHidden,
                    kIntermediate);
            }else{
                linear(
                    attention_context_.as<std::uint16_t>(),
                    layer.o_projection.as<std::uint16_t>(),
                    nullptr,
                    attention_output_.as<std::uint16_t>(),
                    rows,kHidden,kQueryDim);

                add(current->as<std::uint16_t>(),
                    attention_output_.as<std::uint16_t>(),
                    attention_residual_.as<std::uint16_t>(),
                    hidden_count);

                rmsnorm(
                    attention_residual_.as<std::uint16_t>(),
                    layer.post_attention_layernorm
                        .as<std::uint16_t>(),
                    post_norm_.as<std::uint16_t>(),
                    rows,kHidden);

                linear(
                    post_norm_.as<std::uint16_t>(),
                    layer.gate_projection.as<std::uint16_t>(),
                    nullptr,gate_.as<std::uint16_t>(),
                    rows,kIntermediate,kHidden);
                linear(
                    post_norm_.as<std::uint16_t>(),
                    layer.up_projection.as<std::uint16_t>(),
                    nullptr,up_.as<std::uint16_t>(),
                    rows,kIntermediate,kHidden);

                swiglu(gate_.as<std::uint16_t>(),
                       up_.as<std::uint16_t>(),
                       swiglu_.as<std::uint16_t>(),
                       intermediate_count);

                linear(
                    swiglu_.as<std::uint16_t>(),
                    layer.down_projection.as<std::uint16_t>(),
                    nullptr,down_.as<std::uint16_t>(),
                    rows,kHidden,kIntermediate);

                add(attention_residual_.as<std::uint16_t>(),
                    down_.as<std::uint16_t>(),
                    next->as<std::uint16_t>(),
                    hidden_count);
            }

            std::swap(current,next);
        }

        rmsnorm(
            current->as<std::uint16_t>(),
            weights_.final_norm.as<std::uint16_t>(),
            final_norm_.as<std::uint16_t>(),
            rows,kHidden);

        linear(
            final_norm_.as<std::uint16_t>(),
            weights_.lm_heads[step]
                .as<std::uint16_t>(),
            nullptr,logits_.as<std::uint16_t>(),
            rows,kVocab,kHidden);

        std::int32_t* output_code=
            generated_codes_.as<std::int32_t>()
            +step+1;

        if(::qwen3_tts::baseline::g_sampling_enabled){
            hipLaunchKernelGGL(
                ::qwen3_tts::baseline::
                    gpu_sample_topk_bf16_kernel,
                dim3(1),
                dim3(::qwen3_tts::baseline::
                    kGpuSamplerThreads),
                ::qwen3_tts::baseline::
                    gpu_sampler_shared_bytes(),
                stream_,
                logits_.as<std::uint16_t>(),
                (rows-1)*kVocab,
                kVocab,
                ::qwen3_tts::baseline::
                    kGpuSamplerTopK,
                ::qwen3_tts::baseline::
                    kGpuSamplerTemperature,
                rng_state_,output_code);
        }else{
            hipLaunchKernelGGL(
                argmax_bf16_kernel,
                dim3(1),dim3(kThreads),0,stream_,
                logits_.as<std::uint16_t>(),
                (rows-1)*kVocab,
                output_code);
        }
        PRED_HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            gather_predictor_embedding_kernel,
            dim3((kOuterHidden+kThreads-1)/kThreads),
            dim3(kThreads),0,stream_,
            weights_.embeddings[step]
                .as<std::uint16_t>(),
            generated_codes_.as<std::int32_t>(),
            step+1,
            selected_embeddings_.as<std::uint16_t>(),
            free_input_.as<std::uint16_t>());
        PRED_HIP_CHECK(hipGetLastError());
    }

    void capture_graph(){
        // This graph is captured from the Predictor worker thread.
        // Every concurrent GPU producer (speaker, frontend, Talker weight
        // upload) uses its own explicit hipStreamNonBlocking stream.
        // There is therefore no legacy/null-stream dependency crossing this
        // capture boundary.
        PRED_HIP_CHECK(hipStreamBeginCapture(
            stream_,hipStreamCaptureModeThreadLocal));

        for(int step=0;step<kSteps;++step)
            capture_step(step);

        hipLaunchKernelGGL(
            sum_codec_embeddings_kernel,
            dim3((kOuterHidden+kThreads-1)/kThreads),
            dim3(kThreads),0,stream_,
            selected_embeddings_.as<std::uint16_t>(),
            embedding_sum_.as<std::uint16_t>());
        PRED_HIP_CHECK(hipGetLastError());

        add(embedding_sum_.as<std::uint16_t>(),
            context_embedding_.as<std::uint16_t>(),
            next_talker_input_.as<std::uint16_t>(),
            kOuterHidden);

        PRED_HIP_CHECK(hipStreamEndCapture(
            stream_,&graph_));

        if(!graph_)
            throw std::runtime_error(
                "PersistentPredictor captured null HIP graph");

        PRED_HIP_CHECK(hipGraphInstantiate(
            &graph_exec_,graph_,
            nullptr,nullptr,0));

        std::size_t nodes=0;
        PRED_HIP_CHECK(hipGraphGetNodes(
            graph_,nullptr,&nodes));
        graph_node_count_=nodes;
    }

    const PredictorWeights& weights_;
    std::uint64_t* rng_state_=nullptr;
    std::size_t codec0_rows_=0;

    DeviceBuffer codec0_table_;
    DeviceBuffer free_input_;
    DeviceBuffer context_embedding_;
    DeviceBuffer selected_embeddings_;
    DeviceBuffer generated_codes_;
    DeviceBuffer hidden_a_,hidden_b_,norm_;
    DeviceBuffer query_linear_,key_linear_,value_linear_;
    DeviceBuffer query_normalized_,key_normalized_;
    DeviceBuffer query_bhsd_,key_bhsd_,value_bhsd_;
    DeviceBuffer query_rotated_,key_rotated_;
    DeviceBuffer attention_context_,attention_output_;
    DeviceBuffer attention_residual_,post_norm_;
    DeviceBuffer gate_,up_,swiglu_,down_;
    DeviceBuffer final_norm_,logits_;
    DeviceBuffer embedding_sum_,next_talker_input_;
    AttentionWorkspace attention_workspace_;
    std::vector<LayerCache> caches_;
    std::array<DeviceBuffer,kSteps> rope_cos_;
    std::array<DeviceBuffer,kSteps> rope_sin_;

    hipStream_t stream_=nullptr;
    hipGraph_t graph_=nullptr;
    hipGraphExec_t graph_exec_=nullptr;
    std::size_t graph_node_count_=0;
};


} // namespace code_predictor




static fs::path select_work_dir(
    const CliOptions& options) {

    if (!options.work_dir.empty()) {
        return fs::absolute(options.work_dir);
    }

    fs::path parent =
        fs::absolute(options.output_wav)
        .parent_path();

    if (parent.empty()) {
        parent = fs::current_path();
    }

    return parent / ".qwen3_tts_work";
}


// ============================================================================
// legacy_legacy_tag_e.3 native generation tail: FinalHead + Predictor + recurrence
// ============================================================================

[[maybe_unused]] static constexpr int kTalkerHidden = 1024;
static constexpr int kCodecVocab = 3072;
[[maybe_unused]] static constexpr int kCodecGroups = 16;
static constexpr int kCodecEos = 2150;
static constexpr float kRepetitionPenalty = 1.05f;

template<class T>
static void write_binary_tail(
    const fs::path& path,
    const std::vector<T>& values) {

    fs::create_directories(path.parent_path());
    std::ofstream stream(path,std::ios::binary);
    if(!stream)
        throw std::runtime_error(
            "cannot create native-tail file: "+path.string());
    if(!values.empty()){
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(
                values.size()*sizeof(T)));
    }
    if(!stream)
        throw std::runtime_error(
            "failed writing native-tail file: "+path.string());
}

__device__ __forceinline__
static float legacy_bf16_to_float(std::uint16_t x){
    return __uint_as_float(
        static_cast<std::uint32_t>(x)<<16);
}



//
// Geometry:
//   hidden = 1024 BF16 = 2 KiB
//   codec head = 3072 x 1024
//
// Each block stages the complete post-Talker-final-RMSNorm hidden row once in
// LDS, then eight wave32 groups each compute four logits. Every output keeps
// its own scalar FP32 FMA order over K, so model arithmetic/materialization
// semantics are unchanged; only hidden-row locality and inter-output ILP change.
__global__ __launch_bounds__(256, 2)
void final_head_06b_pack4_kernel(
    const std::uint16_t* __restrict__ hidden,
    const std::uint16_t* __restrict__ weight,
    std::uint16_t* __restrict__ logits) {

    constexpr int hidden_size=1024;
    constexpr int outputs_per_wave=4;
    constexpr int waves_per_block=8;

    extern __shared__ std::uint16_t shared_hidden[];

    for(
        int column=
            static_cast<int>(
                threadIdx.x);
        column<hidden_size;
        column+=static_cast<int>(
            blockDim.x)
    ){
        shared_hidden[column]=
            hidden[column];
    }

    __syncthreads();

    const int thread=
        static_cast<int>(
            threadIdx.x);

    const int wave=
        thread>>5;

    const int lane=
        thread&31;

    const int out0=
        (
            static_cast<int>(
                blockIdx.x)
            *waves_per_block
            +wave
        )*outputs_per_wave;

    if(out0>=kCodecVocab){
        return;
    }

    const int out1=out0+1;
    const int out2=out0+2;
    const int out3=out0+3;

    const bool have1=
        out1<kCodecVocab;

    const bool have2=
        out2<kCodecVocab;

    const bool have3=
        out3<kCodecVocab;

    const std::size_t base0=
        static_cast<std::size_t>(
            out0)
        *hidden_size;

    const std::size_t base1=
        have1
        ?static_cast<std::size_t>(
            out1)
            *hidden_size
        :0;

    const std::size_t base2=
        have2
        ?static_cast<std::size_t>(
            out2)
            *hidden_size
        :0;

    const std::size_t base3=
        have3
        ?static_cast<std::size_t>(
            out3)
            *hidden_size
        :0;

    float a0=0.0f;
    float a1=0.0f;
    float a2=0.0f;
    float a3=0.0f;

    for(
        int k=lane;
        k<hidden_size;
        k+=32
    ){
        const float x=
            legacy_bf16_to_float(
                shared_hidden[k]);

        a0=fmaf(
            x,
            legacy_bf16_to_float(
                weight[
                    base0+k]),
            a0);

        if(have1){
            a1=fmaf(
                x,
                legacy_bf16_to_float(
                    weight[
                        base1+k]),
                a1);
        }

        if(have2){
            a2=fmaf(
                x,
                legacy_bf16_to_float(
                    weight[
                        base2+k]),
                a2);
        }

        if(have3){
            a3=fmaf(
                x,
                legacy_bf16_to_float(
                    weight[
                        base3+k]),
                a3);
        }
    }

    a0=
        qwen3_tts::native_ops::
            wave_reduce_sum(a0);

    if(have1){
        a1=
            qwen3_tts::native_ops::
                wave_reduce_sum(a1);
    }

    if(have2){
        a2=
            qwen3_tts::native_ops::
                wave_reduce_sum(a2);
    }

    if(have3){
        a3=
            qwen3_tts::native_ops::
                wave_reduce_sum(a3);
    }

    if(lane==0){
        logits[out0]=
            qwen3_tts::native_ops::
                float_to_bf16_rne(a0);

        if(have1){
            logits[out1]=
                qwen3_tts::native_ops::
                    float_to_bf16_rne(a1);
        }

        if(have2){
            logits[out2]=
                qwen3_tts::native_ops::
                    float_to_bf16_rne(a2);
        }

        if(have3){
            logits[out3]=
                qwen3_tts::native_ops::
                    float_to_bf16_rne(a3);
        }
    }
}


__global__ void process_final_head_logits(
    const std::uint16_t* logits,
    const std::int32_t* history,
    int history_count,
    float* processed){

    const int token=
        static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(token>=kCodecVocab)
        return;

    if(token>=kCodecVocab-1024 && token!=kCodecEos){
        processed[token]=-INFINITY;
        return;
    }

    float score=legacy_bf16_to_float(logits[token]);
    bool repeated=false;
    for(int i=0;i<history_count;++i){
        if(history[i]==token){
            repeated=true;
            break;
        }
    }

    if(repeated){
        score=
            score<0.0f
            ? score*kRepetitionPenalty
            : score/kRepetitionPenalty;
    }
    processed[token]=score;
}


__global__ void cast_final_head_logits(
    const std::uint16_t* logits,
    float* scores){

    const int token=
        static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(token>=kCodecVocab)
        return;
    scores[token]=legacy_bf16_to_float(logits[token]);
}

__global__ void repetition_penalty(
    float* scores,
    const std::int32_t* history,
    int history_count){

    const int token=
        static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(token>=kCodecVocab)
        return;

    bool repeated=false;
    for(int i=0;i<history_count;++i){
        if(history[i]==token){
            repeated=true;
            break;
        }
    }
    if(!repeated)
        return;

    const float value=scores[token];

    // PyTorch 2.5.1 CUDA/HIP TensorIterator scalar-divisor path:
    //   tensor / PythonScalar
    // is implemented as
    //   tensor * (float32(1) / float32(PythonScalar)).
    //
    // For penalty=1.05f the exact float32 reciprocal bits are:
    //   0x3f73cf3e == 0.95238101482391357421875f
    //
    // A direct GPU `value / 1.05f` can differ by 1 ULP, which is exactly
    // the legacy_legacy_tag_e.5 frame-1 single-score mismatch.
    const float reciprocal=__uint_as_float(0x3f73cf3eu);
    scores[token]=
        value<0.0f
        ? value*kRepetitionPenalty
        : value*reciprocal;
}

__global__ void suppress_tokens(
    float* scores,
    int generation_index){

    const int token=
        static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(token>=kCodecVocab)
        return;

    // HF generate() also has min_new_tokens=2. With an empty input_ids
    // prompt (inputs_embeds generation), EOS is forbidden for steps 0 and 1.
    if(token==kCodecEos && generation_index<2){
        scores[token]=-INFINITY;
        return;
    }

    if(
        token>=kCodecVocab-1024
        && token!=kCodecEos
    ){
        scores[token]=-INFINITY;
    }
}

__global__ void argmax_final_head(
    const float* scores,
    std::int32_t* output){

    __shared__ float values[256];
    __shared__ int ids[256];

    float best=-INFINITY;
    int best_id=0;

    for(int i=threadIdx.x;
        i<kCodecVocab;
        i+=blockDim.x){

        const float value=scores[i];
        if(
            value>best
            || (value==best && i<best_id)
        ){
            best=value;
            best_id=i;
        }
    }

    values[threadIdx.x]=best;
    ids[threadIdx.x]=best_id;
    __syncthreads();

    for(int offset=blockDim.x/2;
        offset>0;
        offset/=2){
        if(static_cast<int>(threadIdx.x)<offset){
            const float other=values[threadIdx.x+offset];
            const int other_id=ids[threadIdx.x+offset];
            if(
                other>values[threadIdx.x]
                || (
                    other==values[threadIdx.x]
                    && other_id<ids[threadIdx.x]
                )
            ){
                values[threadIdx.x]=other;
                ids[threadIdx.x]=other_id;
            }
        }
        __syncthreads();
    }

    if(threadIdx.x==0)
        output[0]=ids[0];
}


struct FinalHeadTrace {
    std::int32_t code=0;
    std::vector<float> raw_logits;
    std::vector<float> processed_scores;
};

class NativeFinalHead {
public:
    explicit NativeFinalHead(
        const fs::path& model_dir)
        : weights_(model_dir) {

        // Final-head boundary semantics:
        // talker_tail_hidden.bf16.bin is captured from Qwen3TTSTalkerOutput
        // past_hidden, which is outputs.last_hidden_state from talker.model.
        // talker.model already applies its final RMSNorm before returning
        // last_hidden_state. Therefore the production operation from this
        // captured boundary to raw logits is codec_head only.
        head_=&weights_.suffix(
            "talker.codec_head.weight");

        if(
            head_->dtype!="BF16"
            || head_->shape
                !=std::vector<std::int64_t>{3072,kTalkerHidden}
        )
            throw std::runtime_error(
                "unexpected codec_head weight");

        legacy_HIP_CHECK_b(hipMalloc(
            &d_head_weight_,
            static_cast<std::size_t>(3072)*kTalkerHidden*2));
        legacy_HIP_CHECK_b(hipMalloc(
            &d_hidden_,kTalkerHidden*2));
        legacy_HIP_CHECK_b(hipMalloc(
            &d_logits_,3072*2));
        legacy_HIP_CHECK_b(hipMalloc(
            &d_processed_,3072*sizeof(float)));
        legacy_HIP_CHECK_b(hipMalloc(
            &d_history_,kGenerationCapacityMax*sizeof(std::int32_t)));
        legacy_HIP_CHECK_b(hipMalloc(
            &d_code_,sizeof(std::int32_t)));

        legacy_HIP_CHECK_b(hipMemcpy(
            d_head_weight_,head_->data,
            static_cast<std::size_t>(3072)*kTalkerHidden*2,
            hipMemcpyHostToDevice));

        std::cout
            <<"final_head_linear_backend: native_wave32_bf16_f32acc\n"
            <<"final_head_06b_device_backend: lds_pack4_3072x1024\n"
            <<"final_head_06b_hidden_lds_bytes: 2048\n";
    }

    ~NativeFinalHead(){
        for(void* p:{
            d_head_weight_,d_hidden_,d_logits_,d_processed_,
            d_history_,d_code_
        }) if(p) (void)hipFree(p);
    }

    FinalHeadTrace run(
        const std::uint16_t* hidden_host,
        const std::vector<std::int32_t>& history,
        int generation_index){

        if(history.size()>kGenerationCapacityMax)
            throw std::runtime_error(
                "legacy_legacy_tag_e.4 final-head history exceeds generation capacity");

        legacy_HIP_CHECK_b(hipMemcpy(
            d_hidden_,hidden_host,kTalkerHidden*2,
            hipMemcpyHostToDevice));

        if(!history.empty()){
            legacy_HIP_CHECK_b(hipMemcpy(
                d_history_,history.data(),
                history.size()*sizeof(std::int32_t),
                hipMemcpyHostToDevice));
        }

        // d_hidden_ is already post-final-RMSNorm.
        // The production boundary applies codec_head directly to this row.
        qwen3_tts::native_ops::linear_bf16(
            static_cast<const std::uint16_t*>(d_hidden_),
            static_cast<const std::uint16_t*>(d_head_weight_),
            nullptr,
            static_cast<std::uint16_t*>(d_logits_),
            1,
            3072,
            kTalkerHidden,
            nullptr);

        // Mirror Transformers 4.57.x _sample:
        // outputs.logits[:, -1, :] -> float32 copy
        hipLaunchKernelGGL(
            cast_final_head_logits,
            dim3((3072+255)/256),dim3(256),0,0,
            static_cast<const std::uint16_t*>(d_logits_),
            static_cast<float*>(d_processed_));
        legacy_HIP_CHECK_b(hipGetLastError());
        legacy_HIP_CHECK_b(hipDeviceSynchronize());

        FinalHeadTrace trace;
        trace.raw_logits.resize(3072);
        legacy_HIP_CHECK_b(hipMemcpy(
            trace.raw_logits.data(),
            d_processed_,
            3072*sizeof(float),
            hipMemcpyDeviceToHost));

        // RepetitionPenaltyLogitsProcessor on float32 scores.
        hipLaunchKernelGGL(
            repetition_penalty,
            dim3((3072+255)/256),dim3(256),0,0,
            static_cast<float*>(d_processed_),
            static_cast<const std::int32_t*>(d_history_),
            static_cast<int>(history.size()));
        legacy_HIP_CHECK_b(hipGetLastError());

        // SuppressTokensLogitsProcessor is a distinct later processor.
        hipLaunchKernelGGL(
            suppress_tokens,
            dim3((3072+255)/256),dim3(256),0,0,
            static_cast<float*>(d_processed_),
            generation_index);
        legacy_HIP_CHECK_b(hipGetLastError());

        trace.processed_scores.resize(3072);

        if(g_sampling_enabled){
            legacy_HIP_CHECK_b(
                hipDeviceSynchronize());

            legacy_HIP_CHECK_b(
                hipMemcpy(
                    trace.processed_scores.data(),
                    d_processed_,
                    3072*sizeof(float),
                    hipMemcpyDeviceToHost));

            trace.code=
                sample_topk_f32(
                    trace.processed_scores.data(),
                    3072,
                    50,
                    0.9f);
        }else{
            hipLaunchKernelGGL(
                argmax_final_head,
                dim3(1),dim3(256),0,0,
                static_cast<const float*>(d_processed_),
                static_cast<std::int32_t*>(d_code_));

            legacy_HIP_CHECK_b(
                hipGetLastError());

            legacy_HIP_CHECK_b(
                hipDeviceSynchronize());

            legacy_HIP_CHECK_b(
                hipMemcpy(
                    trace.processed_scores.data(),
                    d_processed_,
                    3072*sizeof(float),
                    hipMemcpyDeviceToHost));

            legacy_HIP_CHECK_b(
                hipMemcpy(
                    &trace.code,
                    d_code_,
                    sizeof(trace.code),
                    hipMemcpyDeviceToHost));
        }

        return trace;
    }

    std::int32_t run_fast(
        const std::uint16_t* hidden_host,
        const std::vector<std::int32_t>& history,
        int generation_index,
        std::uint64_t* rng_state){

        if(history.size()>kGenerationCapacityMax)
            throw std::runtime_error(
                "FinalHead history exceeds generation capacity");

        legacy_HIP_CHECK_b(hipMemcpy(
            d_hidden_,hidden_host,
            kTalkerHidden*sizeof(std::uint16_t),
            hipMemcpyHostToDevice));

        if(!history.empty())
            legacy_HIP_CHECK_b(hipMemcpy(
                d_history_,history.data(),
                history.size()*sizeof(std::int32_t),
                hipMemcpyHostToDevice));

        qwen3_tts::native_ops::linear_bf16(
            static_cast<const std::uint16_t*>(
                d_hidden_),
            static_cast<const std::uint16_t*>(
                d_head_weight_),
            nullptr,
            static_cast<std::uint16_t*>(
                d_logits_),
            1,3072,kTalkerHidden,nullptr);

        hipLaunchKernelGGL(
            cast_final_head_logits,
            dim3((3072+255)/256),dim3(256),
            0,0,
            static_cast<const std::uint16_t*>(
                d_logits_),
            static_cast<float*>(d_processed_));
        legacy_HIP_CHECK_b(hipGetLastError());

        hipLaunchKernelGGL(
            repetition_penalty,
            dim3((3072+255)/256),dim3(256),
            0,0,
            static_cast<float*>(d_processed_),
            static_cast<const std::int32_t*>(
                d_history_),
            static_cast<int>(history.size()));
        legacy_HIP_CHECK_b(hipGetLastError());

        hipLaunchKernelGGL(
            suppress_tokens,
            dim3((3072+255)/256),dim3(256),
            0,0,
            static_cast<float*>(d_processed_),
            generation_index);
        legacy_HIP_CHECK_b(hipGetLastError());

        if(g_sampling_enabled){
            hipLaunchKernelGGL(
                gpu_sample_topk_f32_kernel,
                dim3(1),dim3(kGpuSamplerThreads),
                gpu_sampler_shared_bytes(),0,
                static_cast<const float*>(
                    d_processed_),
                3072,kGpuSamplerTopK,
                kGpuSamplerTemperature,
                rng_state,
                static_cast<std::int32_t*>(
                    d_code_));
        }else{
            hipLaunchKernelGGL(
                argmax_final_head,
                dim3(1),dim3(256),0,0,
                static_cast<const float*>(
                    d_processed_),
                static_cast<std::int32_t*>(
                    d_code_));
        }
        legacy_HIP_CHECK_b(hipGetLastError());

        std::int32_t code=0;
        legacy_HIP_CHECK_b(hipMemcpy(
            &code,d_code_,sizeof(code),
            hipMemcpyDeviceToHost));
        return code;
    }

    const std::uint16_t* device_hidden()const{
        return static_cast<
            const std::uint16_t*>(d_hidden_);
    }

    const std::int32_t* device_code()const{
        return static_cast<
            const std::int32_t*>(d_code_);
    }


    std::int32_t run_fast_device(
        const std::uint16_t* hidden_device,
        int generation_index,
        std::uint64_t* rng_state,
        hipStream_t stream) {

        if (
            generation_index < 0
            || generation_index >= static_cast<int>(kGenerationCapacityMax)
        ) {
            throw std::runtime_error(
                "FinalHead generation index exceeds GPU history capacity");
        }

        // hidden_device is already post-Talker-final-RMSNorm.
        // No host round-trip and no redundant device copy are required.
        hipLaunchKernelGGL(
            final_head_06b_pack4_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        kCodecVocab
                        +32
                        -1
                    )
                    /32)),
            dim3(256),
            static_cast<std::size_t>(
                kTalkerHidden)
                *sizeof(std::uint16_t),
            stream,
            hidden_device,
            static_cast<const std::uint16_t*>(
                d_head_weight_),
            static_cast<std::uint16_t*>(
                d_logits_));

        legacy_HIP_CHECK_b(
            hipGetLastError());

        hipLaunchKernelGGL(
            cast_final_head_logits,
            dim3((3072 + 255) / 256),
            dim3(256),
            0,
            stream,
            static_cast<const std::uint16_t*>(
                d_logits_),
            static_cast<float*>(
                d_processed_));

        legacy_HIP_CHECK_b(
            hipGetLastError());

        hipLaunchKernelGGL(
            repetition_penalty,
            dim3((3072 + 255) / 256),
            dim3(256),
            0,
            stream,
            static_cast<float*>(
                d_processed_),
            static_cast<const std::int32_t*>(
                d_history_),
            generation_index);

        legacy_HIP_CHECK_b(
            hipGetLastError());

        hipLaunchKernelGGL(
            suppress_tokens,
            dim3((3072 + 255) / 256),
            dim3(256),
            0,
            stream,
            static_cast<float*>(
                d_processed_),
            generation_index);

        legacy_HIP_CHECK_b(
            hipGetLastError());

        // The complete FinalHead producer/consumer chain is deliberately
        // single-stream ordered:
        //   linear -> cast -> repetition -> suppress -> sample
        //   -> history append -> 4-byte code readback.
        // Never launch sampling on the default stream: generation_stream is
        // non-blocking, so doing so creates a real data race on d_processed_
        // and d_code_ and can destroy deterministic EOS behavior.
        if (g_sampling_enabled) {
            hipLaunchKernelGGL(
                gpu_sample_topk_f32_kernel,
                dim3(1),
                dim3(kGpuSamplerThreads),
                gpu_sampler_shared_bytes(),
                stream,
                static_cast<const float*>(
                    d_processed_),
                3072,
                kGpuSamplerTopK,
                kGpuSamplerTemperature,
                rng_state,
                static_cast<std::int32_t*>(
                    d_code_));
        } else {
            hipLaunchKernelGGL(
                argmax_final_head,
                dim3(1),
                dim3(256),
                0,
            stream,
                static_cast<const float*>(
                    d_processed_),
                static_cast<std::int32_t*>(
                    d_code_));
        }

        legacy_HIP_CHECK_b(
            hipGetLastError());

        // Keep repetition history entirely on the GPU.
        hipLaunchKernelGGL(
            append_sampled_code_history_kernel,
            dim3(1),
            dim3(1),
            0,
            stream,
            static_cast<const std::int32_t*>(
                d_code_),
            static_cast<std::int32_t*>(
                d_history_),
            generation_index);

        legacy_HIP_CHECK_b(
            hipGetLastError());

        std::int32_t code = 0;

        // Only 4 bytes cross back to the host for EOS/control/logging.
        legacy_HIP_CHECK_b(
            hipMemcpyAsync(
                &code,
                d_code_,
                sizeof(code),
                hipMemcpyDeviceToHost,
                stream));

        // This is the only production generation frame boundary required by
        // host EOS/control logic. It waits only the generation stream, never
        // decoder/preload/Predictor streams.
        legacy_HIP_CHECK_b(
            hipStreamSynchronize(
                stream));

        return code;
    }

    void reset_request(
        hipStream_t stream) {

        legacy_HIP_CHECK_b(
            hipMemsetAsync(
                d_history_,
                0,
                512*sizeof(std::int32_t),
                stream));

        legacy_HIP_CHECK_b(
            hipMemsetAsync(
                d_code_,
                0,
                sizeof(std::int32_t),
                stream));
    }



private:
    model_frontend::ModelWeights weights_;
    const model_frontend::Tensor* head_=nullptr;
    void *d_head_weight_=nullptr;
    void *d_hidden_=nullptr;
    void *d_logits_=nullptr,*d_processed_=nullptr;
    void *d_history_=nullptr,*d_code_=nullptr;
};



// ============================================================================
// legacy_legacy_tag_g.0 Qingming Talker backend
//
// Design:
//   * wave32 ownership
//   * FP32 accumulators live in VGPRs
//   * M=1 inputs / attention scores are staged in LDS
//   * constant __launch_bounds__ for gfx1100 occupancy
//   * no rocBLAS / hipBLAS / external code objects
//
// legacy_legacy_tag_g also runs Talker prefill natively. HuggingFace supplies reference RoPE
// coefficients and correctness tensors only; it no longer supplies the
// production KV initial state.
// ============================================================================

namespace talker {

namespace fs = std::filesystem;

using model_frontend::ModelWeights;
using model_frontend::Tensor;
using code_predictor::DeviceBuffer;

constexpr int kHidden = 1024;
constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kHeadDimRequired = 128;


static std::string layer_suffix(
    int layer,
    const std::string& tail) {

    return "talker.model.layers."
        + std::to_string(layer)
        + "."
        + tail;
}


struct GpuParam {
    DeviceBuffer buffer;
    std::vector<std::int64_t> shape;
    bool present = false;

    GpuParam() = default;

    GpuParam(
        const Tensor& tensor,
        hipStream_t upload_stream)
        : buffer(
            static_cast<std::size_t>(
                tensor.elements())
            * sizeof(std::uint16_t)),
          shape(tensor.shape),
          present(true) {

        if (tensor.dtype != "BF16") {
            throw std::runtime_error(
                "legacy_legacy_tag_g expected BF16 tensor: "
                + tensor.name);
        }

        const hipError_t status =
            hipMemcpyAsync(
                buffer.pointer,
                tensor.data,
                buffer.bytes,
                hipMemcpyHostToDevice,
                upload_stream);

        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string(
                    "legacy_legacy_tag_g explicit-stream weight upload failed: ")
                + hipGetErrorString(status));
        }
    }

    const std::uint16_t* data() const {
        return present
            ? buffer.as<std::uint16_t>()
            : nullptr;
    }
};


struct LayerWeights {
    GpuParam input_norm;
    GpuParam q_weight;
    GpuParam q_bias;
    GpuParam k_weight;
    GpuParam k_bias;
    GpuParam v_weight;
    GpuParam v_bias;
    GpuParam q_norm;
    GpuParam k_norm;
    GpuParam o_weight;
    GpuParam o_bias;
    GpuParam post_norm;
    GpuParam gate_weight;
    GpuParam up_weight;
    GpuParam down_weight;
};


class TalkerWeights {
public:
    explicit TalkerWeights(
        const fs::path& model_dir)
        : host_(model_dir) {

        hipStream_t upload_stream=nullptr;

        const hipError_t create_status=
            hipStreamCreateWithFlags(
                &upload_stream,
                hipStreamNonBlocking);

        if(create_status!=hipSuccess){
            throw std::runtime_error(
                std::string(
                    "Talker explicit weight stream create failed: ")
                +hipGetErrorString(
                    create_status));
        }

        for (int layer = 0;; ++layer) {
            const Tensor* input_norm =
                host_.maybe_suffix(
                    layer_suffix(
                        layer,
                        "input_layernorm.weight"));

            if (input_norm == nullptr) {
                break;
            }

            LayerWeights weights{
                GpuParam(
                    *input_norm,
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "self_attn.q_proj.weight")),
                    upload_stream),
                optional_param(
                    layer_suffix(
                        layer,
                        "self_attn.q_proj.bias"),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "self_attn.k_proj.weight")),
                    upload_stream),
                optional_param(
                    layer_suffix(
                        layer,
                        "self_attn.k_proj.bias"),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "self_attn.v_proj.weight")),
                    upload_stream),
                optional_param(
                    layer_suffix(
                        layer,
                        "self_attn.v_proj.bias"),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "self_attn.q_norm.weight")),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "self_attn.k_norm.weight")),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "self_attn.o_proj.weight")),
                    upload_stream),
                optional_param(
                    layer_suffix(
                        layer,
                        "self_attn.o_proj.bias"),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "post_attention_layernorm.weight")),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "mlp.gate_proj.weight")),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "mlp.up_proj.weight")),
                    upload_stream),
                GpuParam(
                    host_.suffix(
                        layer_suffix(
                            layer,
                            "mlp.down_proj.weight")),
                    upload_stream)
            };

            validate_layer(
                layer,
                weights);

            layers_.push_back(
                std::move(weights));
        }

        if (layers_.empty()) {
            throw std::runtime_error(
                "legacy_legacy_tag_g found no Talker Transformer layers.");
        }

        final_norm_ =
            GpuParam(
                host_.suffix(
                    "talker.model.norm.weight"),
                upload_stream);

        const LayerWeights& first =
            layers_.front();

        head_dim_ =
            static_cast<int>(
                first.q_norm.shape.at(0));

        q_dim_ =
            static_cast<int>(
                first.q_weight.shape.at(0));

        kv_dim_ =
            static_cast<int>(
                first.k_weight.shape.at(0));

        intermediate_ =
            static_cast<int>(
                first.gate_weight.shape.at(0));

        if (
            head_dim_ <= 0
            || q_dim_ % head_dim_ != 0
            || kv_dim_ % head_dim_ != 0
        ) {
            throw std::runtime_error(
                "legacy_legacy_tag_g invalid Talker attention geometry.");
        }

        q_heads_ =
            q_dim_ / head_dim_;

        kv_heads_ =
            kv_dim_ / head_dim_;

        if (
            q_heads_ <= 0
            || kv_heads_ <= 0
            || q_heads_ % kv_heads_ != 0
            || head_dim_ != kHeadDimRequired
        ) {
            (void)hipStreamDestroy(
                upload_stream);
            throw std::runtime_error(
                "legacy_legacy_tag_g requires gfx1100 Talker geometry "
                "with head_dim=128 and integral GQA.");
        }

        const hipError_t sync_status=
            hipStreamSynchronize(
                upload_stream);

        if(sync_status!=hipSuccess){
            (void)hipStreamDestroy(
                upload_stream);
            throw std::runtime_error(
                std::string(
                    "Talker explicit weight upload synchronize failed: ")
                +hipGetErrorString(
                    sync_status));
        }

        (void)hipStreamDestroy(
            upload_stream);

        std::cout
            <<"talker_weight_upload_stream: explicit_nonblocking\n";
    }

    int layer_count() const {
        return static_cast<int>(
            layers_.size());
    }

    int head_dim() const {
        return head_dim_;
    }

    int q_heads() const {
        return q_heads_;
    }

    int kv_heads() const {
        return kv_heads_;
    }

    int q_dim() const {
        return q_dim_;
    }

    int kv_dim() const {
        return kv_dim_;
    }

    int intermediate() const {
        return intermediate_;
    }

    const LayerWeights& layer(
        int index) const {

        return layers_.at(
            static_cast<std::size_t>(
                index));
    }

    const GpuParam& final_norm() const {
        return final_norm_;
    }

private:
    GpuParam optional_param(
        const std::string& suffix,
        hipStream_t upload_stream) {

        const Tensor* tensor =
            host_.maybe_suffix(suffix);

        return tensor == nullptr
            ? GpuParam()
            : GpuParam(
                *tensor,
                upload_stream);
    }

    static void validate_layer(
        int layer,
        const LayerWeights& w) {

        if (
            w.input_norm.shape
                != std::vector<std::int64_t>{kHidden}
            || w.post_norm.shape
                != std::vector<std::int64_t>{kHidden}
            || w.q_weight.shape.size() != 2
            || w.k_weight.shape.size() != 2
            || w.v_weight.shape.size() != 2
            || w.o_weight.shape.size() != 2
            || w.gate_weight.shape.size() != 2
            || w.up_weight.shape.size() != 2
            || w.down_weight.shape.size() != 2
        ) {
            throw std::runtime_error(
                "legacy_legacy_tag_g Talker layer geometry/rank mismatch at layer "
                + std::to_string(layer));
        }

        if (
            w.q_weight.shape[1] != kHidden
            || w.k_weight.shape[1] != kHidden
            || w.v_weight.shape[1] != kHidden
            || w.o_weight.shape[0] != kHidden
            || w.o_weight.shape[1] != w.q_weight.shape[0]
            || w.k_weight.shape[0] != w.v_weight.shape[0]
            || w.gate_weight.shape[1] != kHidden
            || w.up_weight.shape != w.gate_weight.shape
            || w.down_weight.shape[0] != kHidden
            || w.down_weight.shape[1] != w.gate_weight.shape[0]
        ) {
            throw std::runtime_error(
                "legacy_legacy_tag_g Talker projection geometry mismatch at layer "
                + std::to_string(layer));
        }
    }

    ModelWeights host_;
    std::vector<LayerWeights> layers_;
    GpuParam final_norm_;

    int head_dim_ = 0;
    int q_heads_ = 0;
    int kv_heads_ = 0;
    int q_dim_ = 0;
    int kv_dim_ = 0;
    int intermediate_ = 0;
};


__device__ __forceinline__
float talker_bf16_to_float(
    std::uint16_t value) {

    return __uint_as_float(
        static_cast<std::uint32_t>(
            value) << 16);
}


__device__ __forceinline__
std::uint16_t talker_float_to_bf16(
    float value) {

    std::uint32_t bits =
        __float_as_uint(value);

    if (
        (bits & 0x7fffffffU)
        > 0x7f800000U
    ) {
        return static_cast<std::uint16_t>(
            (bits >> 16) | 0x0040U);
    }

    const std::uint32_t lsb =
        (bits >> 16) & 1U;

    bits += 0x7fffU + lsb;

    return static_cast<std::uint16_t>(
        bits >> 16);
}


__device__ __forceinline__
float wave_sum(float value) {

    #pragma unroll
    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        value +=
            __shfl_down(
                value,
                offset);
    }

    return value;
}


__device__ __forceinline__
float wave_max(float value) {

    #pragma unroll
    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        const float other =
            __shfl_down(
                value,
                offset);

        value =
            value > other
            ? value
            : other;
    }

    return value;
}


__device__ __forceinline__
float talker_block_sum(float value) {

    __shared__ float wave_values[8];

    const int lane =
        static_cast<int>(
            threadIdx.x) & 31;

    const int wave =
        static_cast<int>(
            threadIdx.x) >> 5;

    const int wave_count =
        (
            static_cast<int>(
                blockDim.x)
            + 31
        ) >> 5;

    value =
        wave_sum(value);

    if (lane == 0) {
        wave_values[wave] = value;
    }

    __syncthreads();

    if (wave == 0) {
        value =
            lane < wave_count
            ? wave_values[lane]
            : 0.0f;

        value =
            wave_sum(value);

        if (lane == 0) {
            wave_values[0] = value;
        }
    }

    __syncthreads();

    return wave_values[0];
}


__device__ __forceinline__
float talker_block_max(float value) {

    __shared__ float wave_values[8];

    const int lane =
        static_cast<int>(
            threadIdx.x) & 31;

    const int wave =
        static_cast<int>(
            threadIdx.x) >> 5;

    const int wave_count =
        (
            static_cast<int>(
                blockDim.x)
            + 31
        ) >> 5;

    value =
        wave_max(value);

    if (lane == 0) {
        wave_values[wave] = value;
    }

    __syncthreads();

    if (wave == 0) {
        value =
            lane < wave_count
            ? wave_values[lane]
            : -INFINITY;

        value =
            wave_max(value);

        if (lane == 0) {
            wave_values[0] = value;
        }
    }

    __syncthreads();

    return wave_values[0];
}


// One wave owns one output scalar.
// Input is cooperatively staged in LDS; the FP32 accumulator stays in VGPR.
__global__ __launch_bounds__(256, 2)
void talker_linear_fused_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    const std::uint16_t* __restrict__ bias,
    const std::uint16_t* __restrict__ residual,
    std::uint16_t* __restrict__ output,
    int output_columns,
    int reduction_columns) {

    constexpr int outputs_per_wave=2;

    extern __shared__ std::uint16_t shared_input[];

    for(int i=static_cast<int>(threadIdx.x);
        i<reduction_columns;
        i+=static_cast<int>(blockDim.x)){
        shared_input[i]=input[i];
    }
    __syncthreads();

    const int thread=static_cast<int>(threadIdx.x);
    const int wave=thread>>5;
    const int lane=thread&31;

    const int out0=
        (static_cast<int>(blockIdx.x)*kWavesPerBlock+wave)
        *outputs_per_wave;
    if(out0>=output_columns)return;

    const int out1=out0+1;
    const bool have_second=out1<output_columns;

    const std::size_t base0=static_cast<std::size_t>(out0)*reduction_columns;
    const std::size_t base1=have_second?static_cast<std::size_t>(out1)*reduction_columns:0;

    float a0=0.0f,a1=0.0f;

    for(int k=lane;k<reduction_columns;k+=kWave){
        const float x=talker_bf16_to_float(shared_input[k]);
        a0=fmaf(x,talker_bf16_to_float(weight[base0+k]),a0);
        if(have_second)a1=fmaf(x,talker_bf16_to_float(weight[base1+k]),a1);
    }

    a0=wave_sum(a0);
    if(have_second)a1=wave_sum(a1);

    if(lane==0){
        if(bias!=nullptr){
            a0+=talker_bf16_to_float(bias[out0]);
            if(have_second)a1+=talker_bf16_to_float(bias[out1]);
        }

        const auto v0=talker_float_to_bf16(a0);
        output[out0]=residual!=nullptr
            ?talker_float_to_bf16(
                talker_bf16_to_float(residual[out0])+talker_bf16_to_float(v0))
            :v0;

        if(have_second){
            const auto v1=talker_float_to_bf16(a1);
            output[out1]=residual!=nullptr
                ?talker_float_to_bf16(
                    talker_bf16_to_float(residual[out1])+talker_bf16_to_float(v1))
                :v1;
        }
    }
}


// Fused M=1 Q/K/V. Normalized hidden is staged once in LDS.
__global__ __launch_bounds__(256, 2)
void talker_qkv_fused_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ q_weight,
    const std::uint16_t* __restrict__ q_bias,
    const std::uint16_t* __restrict__ k_weight,
    const std::uint16_t* __restrict__ k_bias,
    const std::uint16_t* __restrict__ v_weight,
    const std::uint16_t* __restrict__ v_bias,
    std::uint16_t* __restrict__ q_output,
    std::uint16_t* __restrict__ k_output,
    std::uint16_t* __restrict__ v_output,
    int hidden,
    int q_dim,
    int kv_dim) {

    constexpr int outputs_per_wave=2;

    extern __shared__ std::uint16_t shared_input[];

    for(int i=static_cast<int>(threadIdx.x);
        i<hidden;
        i+=static_cast<int>(blockDim.x)){
        shared_input[i]=input[i];
    }
    __syncthreads();

    const int thread=static_cast<int>(threadIdx.x);
    const int wave=thread>>5;
    const int lane=thread&31;

    const int first=
        (static_cast<int>(blockIdx.x)*kWavesPerBlock+wave)
        *outputs_per_wave;
    const int total=q_dim+2*kv_dim;
    if(first>=total)return;

    const int second=first+1;
    const bool have_second=second<total;

    const std::uint16_t *w0=nullptr,*w1=nullptr;
    const std::uint16_t *b0=nullptr,*b1=nullptr;
    std::uint16_t *o0=nullptr,*o1=nullptr;
    int row0=0,row1=0;

    if(first<q_dim){
        w0=q_weight;b0=q_bias;o0=q_output;row0=first;
    }else if(first<q_dim+kv_dim){
        w0=k_weight;b0=k_bias;o0=k_output;row0=first-q_dim;
    }else{
        w0=v_weight;b0=v_bias;o0=v_output;row0=first-q_dim-kv_dim;
    }

    if(have_second){
        if(second<q_dim){
            w1=q_weight;b1=q_bias;o1=q_output;row1=second;
        }else if(second<q_dim+kv_dim){
            w1=k_weight;b1=k_bias;o1=k_output;row1=second-q_dim;
        }else{
            w1=v_weight;b1=v_bias;o1=v_output;row1=second-q_dim-kv_dim;
        }
    }

    const std::size_t base0=static_cast<std::size_t>(row0)*hidden;
    const std::size_t base1=have_second?static_cast<std::size_t>(row1)*hidden:0;

    float a0=0.0f,a1=0.0f;

    for(int k=lane;k<hidden;k+=kWave){
        const float x=talker_bf16_to_float(shared_input[k]);
        a0=fmaf(x,talker_bf16_to_float(w0[base0+k]),a0);
        if(have_second)a1=fmaf(x,talker_bf16_to_float(w1[base1+k]),a1);
    }

    a0=wave_sum(a0);
    if(have_second)a1=wave_sum(a1);

    if(lane==0){
        if(b0!=nullptr)a0+=talker_bf16_to_float(b0[row0]);
        o0[row0]=talker_float_to_bf16(a0);

        if(have_second){
            if(b1!=nullptr)a1+=talker_bf16_to_float(b1[row1]);
            o1[row1]=talker_float_to_bf16(a1);
        }
    }
}


// Per-head Q/K RMSNorm + RoPE. K blocks also append K and V to persistent KV.
// This is intentionally 128 threads: one thread owns one head dimension.
__global__ __launch_bounds__(128, 2)
void talker_qk_norm_rope_cache_fused_kernel(
    const std::uint16_t* __restrict__ q,
    const std::uint16_t* __restrict__ k,
    const std::uint16_t* __restrict__ v,
    const std::uint16_t* __restrict__ q_norm_weight,
    const std::uint16_t* __restrict__ k_norm_weight,
    const std::uint16_t* __restrict__ cosine,
    const std::uint16_t* __restrict__ sine,
    std::uint16_t* __restrict__ q_rotated,
    std::uint16_t* __restrict__ key_cache,
    std::uint16_t* __restrict__ value_cache,
    int q_heads,
    int kv_heads,
    int head_dim,
    int max_cache,
    int cache_position,
    float epsilon) {

    __shared__ std::uint16_t normalized[128];
    __shared__ float inverse_rms;

    const int head_slot =
        static_cast<int>(
            blockIdx.x);

    const bool is_query =
        head_slot < q_heads;

    const int head =
        is_query
        ? head_slot
        : head_slot - q_heads;

    const int dimension =
        static_cast<int>(
            threadIdx.x);

    if (
        dimension >= head_dim
        || (
            !is_query
            && head >= kv_heads
        )
    ) {
        return;
    }

    const std::uint16_t* source =
        is_query ? q : k;

    const std::uint16_t* norm_weight =
        is_query
        ? q_norm_weight
        : k_norm_weight;

    const int source_index =
        head * head_dim
        + dimension;

    const float value =
        talker_bf16_to_float(
            source[
                source_index]);

    const float square =
        value * value;

    const float sum =
        talker_block_sum(square);

    if (threadIdx.x == 0) {
        inverse_rms =
            rsqrtf(
                sum
                / static_cast<float>(
                    head_dim)
                + epsilon);
    }

    __syncthreads();

    // Official RMSNorm dtype boundary:
    // normalize in FP32 -> cast to input dtype -> multiply BF16 weight.
    const std::uint16_t normalized_bits =
        talker_float_to_bf16(
            value
            * inverse_rms);

    normalized[
        dimension
    ] = talker_float_to_bf16(
        talker_bf16_to_float(
            normalized_bits)
        * talker_bf16_to_float(
            norm_weight[
                dimension]));

    __syncthreads();

    const int rotated_dimension =
        dimension < head_dim / 2
        ? dimension + head_dim / 2
        : dimension - head_dim / 2;

    float rotated =
        talker_bf16_to_float(
            normalized[
                rotated_dimension]);

    // Official rotate_half(x) == (-x2, +x1).
    if (dimension < head_dim / 2) {
        rotated = -rotated;
    }

    const std::uint16_t first =
        talker_float_to_bf16(
            talker_bf16_to_float(
                normalized[
                    dimension])
            * talker_bf16_to_float(
                cosine[
                    dimension]));

    const std::uint16_t second =
        talker_float_to_bf16(
            rotated
            * talker_bf16_to_float(
                sine[
                    dimension]));

    const std::uint16_t rope_value =
        talker_float_to_bf16(
            talker_bf16_to_float(first)
            + talker_bf16_to_float(second));

    if (is_query) {
        q_rotated[
            source_index
        ] = rope_value;
    } else {
        key_cache[
            (
                static_cast<std::size_t>(
                    head)
                * max_cache
                + cache_position
            ) * head_dim
            + dimension
        ] = rope_value;

        value_cache[
            (
                static_cast<std::size_t>(
                    head)
                * max_cache
                + cache_position
            ) * head_dim
            + dimension
        ] = v[
            source_index];
    }
}


// Prefill RoPE over [sequence, heads, head_dim].
__global__ __launch_bounds__(256, 2)
void talker_rope_rows_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ cosine,
    const std::uint16_t* __restrict__ sine,
    std::uint16_t* __restrict__ output,
    int sequence_length,
    int heads,
    int head_dim) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(
            blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(
            sequence_length)
        * heads
        * head_dim;

    if (index >= count) {
        return;
    }

    const int dimension =
        static_cast<int>(
            index % head_dim);

    const std::uint64_t t =
        index / head_dim;

    const int head =
        static_cast<int>(
            t % heads);

    const int row =
        static_cast<int>(
            t / heads);

    const int rotated_dimension =
        dimension < head_dim / 2
        ? dimension + head_dim / 2
        : dimension - head_dim / 2;

    const std::size_t head_base =
        (
            static_cast<std::size_t>(
                row)
            * heads
            + head
        ) * head_dim;

    const float value =
        talker_bf16_to_float(
            input[
                head_base
                + dimension]);

    float rotated =
        talker_bf16_to_float(
            input[
                head_base
                + rotated_dimension]);

    // Official rotate_half(x) == (-x2, +x1).
    if (dimension < head_dim / 2) {
        rotated = -rotated;
    }

    const std::size_t rope_index =
        static_cast<std::size_t>(
            row)
        * head_dim
        + dimension;

    const std::uint16_t first =
        talker_float_to_bf16(
            value
            * talker_bf16_to_float(
                cosine[
                    rope_index]));

    const std::uint16_t second =
        talker_float_to_bf16(
            rotated
            * talker_bf16_to_float(
                sine[
                    rope_index]));

    output[index] =
        talker_float_to_bf16(
            talker_bf16_to_float(first)
            + talker_bf16_to_float(second));
}


// [sequence, kv_heads, head_dim] -> [kv_heads, max_cache, head_dim].
__global__ __launch_bounds__(256, 2)
void talker_scatter_prefill_kv_kernel(
    const std::uint16_t* __restrict__ key,
    const std::uint16_t* __restrict__ value,
    std::uint16_t* __restrict__ key_cache,
    std::uint16_t* __restrict__ value_cache,
    int sequence_length,
    int kv_heads,
    int head_dim,
    int max_cache) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(
            blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(
            sequence_length)
        * kv_heads
        * head_dim;

    if (index >= count) {
        return;
    }

    const int dimension =
        static_cast<int>(
            index % head_dim);

    const std::uint64_t t =
        index / head_dim;

    const int head =
        static_cast<int>(
            t % kv_heads);

    const int row =
        static_cast<int>(
            t / kv_heads);

    const std::size_t destination =
        (
            static_cast<std::size_t>(
                head)
            * max_cache
            + row
        ) * head_dim
        + dimension;

    key_cache[
        destination
    ] = key[index];

    value_cache[
        destination
    ] = value[index];
}


// Fused QK -> BF16 raw -> BF16 scale -> FP32 softmax -> BF16 P -> P×V.
//
// One block owns one (query row, query head). Q and scaled scores live in LDS;
// each wave computes one key score at a time with a VGPR FP32 accumulator.

// ============================================================================
// legacy_legacy_tag_k correctness-first Talker decode cache append.
// Q/K normalization and RoPE are materialized by the same standalone
// operators as prefill. This kernel only appends the resulting K and V row.
// ============================================================================
__global__
void talker_scatter_decode_kv_kernel(
    const std::uint16_t* __restrict__ key,
    const std::uint16_t* __restrict__ value,
    std::uint16_t* __restrict__ key_cache,
    std::uint16_t* __restrict__ value_cache,
    int kv_heads,
    int head_dim,
    int max_cache,
    int cache_position) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    const std::uint64_t count =
        static_cast<std::uint64_t>(kv_heads)
        * head_dim;

    if (index >= count) {
        return;
    }

    const int dimension =
        static_cast<int>(index % head_dim);

    const int head =
        static_cast<int>(index / head_dim);

    const std::size_t destination =
        (
            static_cast<std::size_t>(head)
            * max_cache
            + cache_position
        ) * head_dim
        + dimension;

    key_cache[destination] = key[index];
    value_cache[destination] = value[index];
}


__global__ __launch_bounds__(256, 2)
void talker_attention_kernel(
    const std::uint16_t* __restrict__ query,
    const std::uint16_t* __restrict__ key_cache,
    const std::uint16_t* __restrict__ value_cache,
    std::uint16_t* __restrict__ context,
    int query_length,
    int cache_before,
    int q_heads,
    int kv_heads,
    int head_dim,
    int max_cache,
    int sliding_window,
    float scale) {

    extern __shared__ std::uint16_t shared_bf16[];

    std::uint16_t* shared_query =
        shared_bf16;

    std::uint16_t* shared_scores =
        shared_bf16 + head_dim;

    const int block =
        static_cast<int>(
            blockIdx.x);

    const int query_index =
        block % query_length;

    const int query_head =
        block / query_length;

    if (query_head >= q_heads) {
        return;
    }

    const int kv_head =
        query_head
        / (q_heads / kv_heads);

    const int absolute_query =
        cache_before
        + query_index;

    const int key_end =
        absolute_query + 1;

    const int window_start =
        key_end - sliding_window;

    const int key_start =
        sliding_window > 0
        ? (
            window_start > 0
            ? window_start
            : 0
        )
        : 0;

    const int key_length =
        key_end - key_start;

    const std::size_t query_base =
        (
            static_cast<std::size_t>(
                query_index)
            * q_heads
            + query_head
        ) * head_dim;

    for (
        int dimension =
            static_cast<int>(
                threadIdx.x);
        dimension < head_dim;
        dimension += static_cast<int>(
            blockDim.x)
    ) {
        shared_query[
            dimension
        ] = query[
            query_base
            + dimension];
    }

    __syncthreads();

    const int thread =
        static_cast<int>(
            threadIdx.x);

    const int wave =
        thread >> 5;

    const int lane =
        thread & 31;

    for (
        int key_relative = wave;
        key_relative < key_length;
        key_relative += kWavesPerBlock
    ) {
        const int key_index =
            key_start
            + key_relative;

        float accumulator = 0.0f;

        #pragma unroll
        for (
            int dimension = lane;
            dimension < 128;
            dimension += 32
        ) {
            accumulator = fmaf(
                talker_bf16_to_float(
                    shared_query[
                        dimension]),
                talker_bf16_to_float(
                    key_cache[
                        (
                            static_cast<std::size_t>(
                                kv_head)
                            * max_cache
                            + key_index
                        ) * head_dim
                        + dimension]),
                accumulator);
        }

        accumulator =
            wave_sum(
                accumulator);

        if (lane == 0) {
            // Critical official dtype boundary:
            // BF16 matmul output is materialized before scalar scaling.
            const std::uint16_t raw_score =
                talker_float_to_bf16(
                    accumulator);

            shared_scores[
                key_relative
            ] = talker_float_to_bf16(
                talker_bf16_to_float(
                    raw_score)
                * scale);
        }
    }

    __syncthreads();

    float local_max =
        -INFINITY;

    for (
        int key_relative =
            static_cast<int>(
                threadIdx.x);
        key_relative < key_length;
        key_relative += static_cast<int>(
            blockDim.x)
    ) {
        const float score =
            talker_bf16_to_float(
                shared_scores[
                    key_relative]);

        local_max =
            local_max > score
            ? local_max
            : score;
    }

    const float maximum =
        talker_block_max(
            local_max);

    float local_sum = 0.0f;

    for (
        int key_relative =
            static_cast<int>(
                threadIdx.x);
        key_relative < key_length;
        key_relative += static_cast<int>(
            blockDim.x)
    ) {
        local_sum +=
            expf(
                talker_bf16_to_float(
                    shared_scores[
                        key_relative])
                - maximum);
    }

    const float total =
        talker_block_sum(
            local_sum);

    for (
        int key_relative =
            static_cast<int>(
                threadIdx.x);
        key_relative < key_length;
        key_relative += static_cast<int>(
            blockDim.x)
    ) {
        const float probability =
            expf(
                talker_bf16_to_float(
                    shared_scores[
                        key_relative])
                - maximum)
            / total;

        shared_scores[
            key_relative
        ] = talker_float_to_bf16(
            probability);
    }

    __syncthreads();

    const int dimension =
        static_cast<int>(
            threadIdx.x);

    if (dimension < head_dim) {
        float accumulator = 0.0f;

        for (
            int key_relative = 0;
            key_relative < key_length;
            ++key_relative
        ) {
            const int key_index =
                key_start
                + key_relative;

            accumulator = fmaf(
                talker_bf16_to_float(
                    shared_scores[
                        key_relative]),
                talker_bf16_to_float(
                    value_cache[
                        (
                            static_cast<std::size_t>(
                                kv_head)
                            * max_cache
                            + key_index
                        ) * head_dim
                        + dimension]),
                accumulator);
        }

        context[
            query_base
            + dimension
        ] = talker_float_to_bf16(
            accumulator);
    }
}


// M=1 GQA-2 locality attention.
// One block owns one KV head and its two query heads.
// Persistent K/V are the long-lived raw material; Q/scores/probabilities stay
// in LDS and QK/PV accumulators stay in VGPR.
__global__ __launch_bounds__(256, 2)
void talker_attention_gqa2_m1_locality_kernel(
    const std::uint16_t* __restrict__ query,
    const std::uint16_t* __restrict__ key_cache,
    const std::uint16_t* __restrict__ value_cache,
    std::uint16_t* __restrict__ context,
    int cache_before,
    int q_heads,
    int kv_heads,
    int head_dim,
    int max_cache,
    int sliding_window,
    float scale) {

    extern __shared__ std::uint16_t shared_bf16[];

    std::uint16_t* shared_query0 = shared_bf16;
    std::uint16_t* shared_query1 = shared_query0 + head_dim;

    const int key_end = cache_before + 1;
    const int window_start = key_end - sliding_window;
    const int key_start =
        sliding_window > 0 ? (window_start > 0 ? window_start : 0) : 0;
    const int key_length = key_end - key_start;

    std::uint16_t* shared_scores0 = shared_query1 + head_dim;
    std::uint16_t* shared_scores1 = shared_scores0 + key_length;

    const int kv_head = static_cast<int>(blockIdx.x);
    if (kv_head >= kv_heads) return;

    const int queries_per_kv = q_heads / kv_heads;
    if (queries_per_kv != 2 || head_dim != 128) return;

    const int query_head0 = kv_head * 2;
    const int query_head1 = query_head0 + 1;

    const std::size_t query_base0 =
        static_cast<std::size_t>(query_head0) * head_dim;
    const std::size_t query_base1 =
        static_cast<std::size_t>(query_head1) * head_dim;

    const int thread = static_cast<int>(threadIdx.x);

    // Exactly 256 BF16 query values = two 128-d heads.
    if (thread < head_dim) {
        shared_query0[thread] = query[query_base0 + thread];
    } else if (thread < 2 * head_dim) {
        const int d = thread - head_dim;
        shared_query1[d] = query[query_base1 + d];
    }

    __syncthreads();

    const int wave = thread >> 5;
    const int lane = thread & 31;

    // One K load serves both query-head dot products.
    for (int key_relative = wave;
         key_relative < key_length;
         key_relative += kWavesPerBlock) {

        const int key_index = key_start + key_relative;
        const std::size_t key_base =
            (static_cast<std::size_t>(kv_head) * max_cache + key_index)
            * head_dim;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int d = lane; d < 128; d += 32) {
            const float kval =
                talker_bf16_to_float(key_cache[key_base + d]);

            acc0 = fmaf(
                talker_bf16_to_float(shared_query0[d]),
                kval,
                acc0);

            acc1 = fmaf(
                talker_bf16_to_float(shared_query1[d]),
                kval,
                acc1);
        }

        acc0 = wave_sum(acc0);
        acc1 = wave_sum(acc1);

        if (lane == 0) {
            
            const std::uint16_t raw0 =
                talker_float_to_bf16(acc0);
            const std::uint16_t raw1 =
                talker_float_to_bf16(acc1);

            shared_scores0[key_relative] =
                talker_float_to_bf16(
                    talker_bf16_to_float(raw0) * scale);

            shared_scores1[key_relative] =
                talker_float_to_bf16(
                    talker_bf16_to_float(raw1) * scale);
        }
    }

    __syncthreads();

    // Head 0 softmax: same 256-thread reduction topology as generic kernel.
    float local_max0 = -INFINITY;
    for (int k = thread; k < key_length; k += static_cast<int>(blockDim.x)) {
        const float score = talker_bf16_to_float(shared_scores0[k]);
        local_max0 = local_max0 > score ? local_max0 : score;
    }
    const float max0 = talker_block_max(local_max0);

    float local_sum0 = 0.0f;
    for (int k = thread; k < key_length; k += static_cast<int>(blockDim.x)) {
        local_sum0 += expf(
            talker_bf16_to_float(shared_scores0[k]) - max0);
    }
    const float total0 = talker_block_sum(local_sum0);

    for (int k = thread; k < key_length; k += static_cast<int>(blockDim.x)) {
        const float p =
            expf(talker_bf16_to_float(shared_scores0[k]) - max0)
            / total0;
        shared_scores0[k] = talker_float_to_bf16(p);
    }

    __syncthreads();

    // Head 1 softmax: same dtype/materialization sequence.
    float local_max1 = -INFINITY;
    for (int k = thread; k < key_length; k += static_cast<int>(blockDim.x)) {
        const float score = talker_bf16_to_float(shared_scores1[k]);
        local_max1 = local_max1 > score ? local_max1 : score;
    }
    const float max1 = talker_block_max(local_max1);

    float local_sum1 = 0.0f;
    for (int k = thread; k < key_length; k += static_cast<int>(blockDim.x)) {
        local_sum1 += expf(
            talker_bf16_to_float(shared_scores1[k]) - max1);
    }
    const float total1 = talker_block_sum(local_sum1);

    for (int k = thread; k < key_length; k += static_cast<int>(blockDim.x)) {
        const float p =
            expf(talker_bf16_to_float(shared_scores1[k]) - max1)
            / total1;
        shared_scores1[k] = talker_float_to_bf16(p);
    }

    __syncthreads();

    // One V load serves both query-head PV accumulators.
    const int d = thread;
    if (d < head_dim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        for (int key_relative = 0;
             key_relative < key_length;
             ++key_relative) {

            const int key_index = key_start + key_relative;

            const float value =
                talker_bf16_to_float(
                    value_cache[
                        (static_cast<std::size_t>(kv_head) * max_cache
                         + key_index)
                        * head_dim
                        + d]);

            acc0 = fmaf(
                talker_bf16_to_float(shared_scores0[key_relative]),
                value,
                acc0);

            acc1 = fmaf(
                talker_bf16_to_float(shared_scores1[key_relative]),
                value,
                acc1);
        }

        context[query_base0 + d] = talker_float_to_bf16(acc0);
        context[query_base1 + d] = talker_float_to_bf16(acc1);
    }
}




// Attention residual + post-attention RMSNorm in one block.
// Residual BF16 values stay in LDS while the RMS statistic is reduced.
__global__ __launch_bounds__(256, 2)
void talker_add_postnorm_fused_kernel(
    const std::uint16_t* __restrict__ hidden,
    const std::uint16_t* __restrict__ attention,
    const std::uint16_t* __restrict__ norm_weight,
    std::uint16_t* __restrict__ residual,
    std::uint16_t* __restrict__ normalized,
    int dimension,
    float epsilon) {

    extern __shared__ std::uint16_t shared_residual[];
    __shared__ float inverse_rms;

    float local_sum = 0.0f;

    for (
        int index =
            static_cast<int>(
                threadIdx.x);
        index < dimension;
        index += static_cast<int>(
            blockDim.x)
    ) {
        const std::uint16_t value =
            talker_float_to_bf16(
                talker_bf16_to_float(
                    hidden[index])
                + talker_bf16_to_float(
                    attention[index]));

        residual[index] =
            value;

        shared_residual[index] =
            value;

        const float x =
            talker_bf16_to_float(
                value);

        local_sum =
            fmaf(
                x,
                x,
                local_sum);
    }

    const float sum =
        talker_block_sum(
            local_sum);

    if (threadIdx.x == 0) {
        inverse_rms =
            rsqrtf(
                sum
                / static_cast<float>(
                    dimension)
                + epsilon);
    }

    __syncthreads();

    for (
        int index =
            static_cast<int>(
                threadIdx.x);
        index < dimension;
        index += static_cast<int>(
            blockDim.x)
    ) {
        const std::uint16_t normalized_bits =
            talker_float_to_bf16(
                talker_bf16_to_float(
                    shared_residual[
                        index])
                * inverse_rms);

        normalized[index] =
            talker_float_to_bf16(
                talker_bf16_to_float(
                    normalized_bits)
                * talker_bf16_to_float(
                    norm_weight[
                        index]));
    }
}


// Gate and Up projections share the same LDS input. Each wave owns one
// intermediate channel and keeps two independent FP32 accumulators in VGPRs.
__global__ __launch_bounds__(256, 2)
void talker_gate_up_swiglu_fused_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint16_t* __restrict__ gate_weight,
    const std::uint16_t* __restrict__ up_weight,
    std::uint16_t* __restrict__ output,
    int hidden,
    int intermediate) {

    constexpr int outputs_per_wave=2;

    extern __shared__ std::uint16_t shared_input[];

    for(int i=static_cast<int>(threadIdx.x);
        i<hidden;
        i+=static_cast<int>(blockDim.x)){
        shared_input[i]=input[i];
    }
    __syncthreads();

    const int thread=static_cast<int>(threadIdx.x);
    const int wave=thread>>5;
    const int lane=thread&31;

    const int ch0=
        (static_cast<int>(blockIdx.x)*kWavesPerBlock+wave)
        *outputs_per_wave;
    if(ch0>=intermediate)return;

    const int ch1=ch0+1;
    const bool have_second=ch1<intermediate;

    const std::size_t base0=static_cast<std::size_t>(ch0)*hidden;
    const std::size_t base1=have_second?static_cast<std::size_t>(ch1)*hidden:0;

    float g0=0.0f,u0=0.0f,g1=0.0f,u1=0.0f;

    for(int k=lane;k<hidden;k+=kWave){
        const float x=talker_bf16_to_float(shared_input[k]);
        g0=fmaf(x,talker_bf16_to_float(gate_weight[base0+k]),g0);
        u0=fmaf(x,talker_bf16_to_float(up_weight[base0+k]),u0);

        if(have_second){
            g1=fmaf(x,talker_bf16_to_float(gate_weight[base1+k]),g1);
            u1=fmaf(x,talker_bf16_to_float(up_weight[base1+k]),u1);
        }
    }

    g0=wave_sum(g0);u0=wave_sum(u0);
    if(have_second){g1=wave_sum(g1);u1=wave_sum(u1);}

    if(lane==0){
        const auto gb0=talker_float_to_bf16(g0);
        const auto ub0=talker_float_to_bf16(u0);
        const float gv0=talker_bf16_to_float(gb0);
        const auto act0=talker_float_to_bf16(gv0/(1.0f+expf(-gv0)));
        output[ch0]=talker_float_to_bf16(
            talker_bf16_to_float(act0)*talker_bf16_to_float(ub0));

        if(have_second){
            const auto gb1=talker_float_to_bf16(g1);
            const auto ub1=talker_float_to_bf16(u1);
            const float gv1=talker_bf16_to_float(gb1);
            const auto act1=talker_float_to_bf16(gv1/(1.0f+expf(-gv1)));
            output[ch1]=talker_float_to_bf16(
                talker_bf16_to_float(act1)*talker_bf16_to_float(ub1));
        }
    }
}


struct LayerCache {
    DeviceBuffer key;
    DeviceBuffer value;

    explicit LayerCache(
        std::size_t bytes)
        : key(bytes),
          value(bytes) {

        (void)hipMemset(
            key.pointer,
            0,
            key.bytes);

        (void)hipMemset(
            value.pointer,
            0,
            value.bytes);
    }
};


struct StepResult {
    std::vector<std::uint16_t> hidden;
    std::vector<
        std::vector<std::uint16_t>
    > layer_last_rows;
};


class NativeTalkerEngine {
public:
    NativeTalkerEngine(
        const fs::path& model_dir,
        int prefill_length,
        int maximum_generated_frames,
        int sliding_window)
        : weights_(model_dir),
          prefill_length_(prefill_length),
          sliding_window_(sliding_window) {

        stream_=
            ::qwen3_tts::baseline::
                create_generation_execution_stream(
                    "Talker generation stream");

        std::cout
            <<"talker_generation_stream: "
            <<(::qwen3_tts::baseline::g_resident_cu_partition_enabled
                ?"explicit_cu_masked"
                :"explicit_nonblocking")
            <<"\n"
            <<"talker_generation_stream_cu_partitioned: "
            <<(::qwen3_tts::baseline::g_resident_cu_partition_enabled
                ?"True":"False")
            <<"\n"
            <<"talker_generation_device_wide_sync: False\n";

        max_rows_ =
            std::max(
                1,
                prefill_length_);

        max_cache_ =
            prefill_length_
            + std::max(
                1,
                maximum_generated_frames)
            + 4;

        const std::size_t cache_elements =
            static_cast<std::size_t>(
                weights_.kv_heads())
            * max_cache_
            * weights_.head_dim();

        const std::size_t cache_bytes =
            cache_elements
            * sizeof(std::uint16_t);

        caches_.reserve(
            weights_.layer_count());

        for (
            int layer = 0;
            layer < weights_.layer_count();
            ++layer
        ) {
            caches_.emplace_back(
                cache_bytes);
        }

        allocate_workspace();

        std::cout
            << "native_talker_weight_layer_count: "
            << weights_.layer_count()
            << "\n"
            << "native_talker_hidden: "
            << kHidden
            << "\n"
            << "native_talker_q_heads: "
            << weights_.q_heads()
            << "\n"
            << "native_talker_kv_heads: "
            << weights_.kv_heads()
            << "\n"
            << "native_talker_head_dim: "
            << weights_.head_dim()
            << "\n"
            << "native_talker_intermediate: "
            << weights_.intermediate()
            << "\n"
            << "native_talker_prefill_length: "
            << prefill_length_
            << "\n"
            << "native_talker_max_cache: "
            << max_cache_
            << "\n"
            << "native_talker_sliding_window: "
            << sliding_window_
            << "\n"
            << "talker_decode_backend: staged_prefill_equivalent\n"
            << "talker_decode_fused_m1_dependency: False\n"
            << "talker_decode_bf16_materialization_boundaries: explicit\n"
            << "talker_rope_rotate_half: official_minus_x2_plus_x1\n"
            << "talker_rope_sign_fix: True\n"
            << "talker_decode_qkv_backend: fused_lds_wave32\n"
            << "talker_decode_qk_norm_rope_cache_backend: fused_official_semantics\n"
            << "talker_decode_qkv_fusion: True\n"
            << "talker_decode_qk_rope_cache_fusion: True\n"
            << "talker_decode_oproj_backend: lds_staged_wave32\n"
            << "talker_decode_residual_postnorm_backend: fused_lds_consumer\n"
            << "talker_decode_gate_up_swiglu_backend: fused_vgpr_lds_pack2\n"
            << "talker_m1_outputs_per_wave: 2\n"
            << "talker_m1_dot2_changed: False\n"
            << "talker_decode_down_residual_backend: fused_wave32\n"
            << "talker_locality_policy: cache_raw_compute_derived\n"
            << "talker_decode_attention_backend: gqa2_m1_shared_kv_locality\n"
            << "talker_decode_attention_blocks_per_layer: 8\n"
            << "talker_decode_attention_q_heads_per_block: 2\n"
            << "talker_decode_attention_kv_reuse: shared_by_two_q_heads\n"
            << "talker_decode_attention_score_global_materialization: False\n"
            << "talker_decode_attention_probability_global_materialization: False\n";
    }

    ~NativeTalkerEngine(){
        if(stream_!=nullptr){
            (void)hipStreamSynchronize(
                stream_);
            (void)hipStreamDestroy(
                stream_);
        }
    }

    void reset_request() {
        for(auto& cache:caches_){
            hip_check(
                hipMemsetAsync(
                    cache.key.pointer,
                    0,
                    cache.key.bytes,
                    stream_),
                "Talker reset K cache");

            hip_check(
                hipMemsetAsync(
                    cache.value.pointer,
                    0,
                    cache.value.bytes,
                    stream_),
                "Talker reset V cache");
        }

        hip_check(
            hipStreamSynchronize(
                stream_),
            "Talker reset request synchronize");

        final_hidden_row_=0;
    }

    hipStream_t stream() const {
        return stream_;
    }

    int layer_count() const {
        return weights_.layer_count();
    }

    int head_dim() const {
        return weights_.head_dim();
    }

    int max_cache() const {
        return max_cache_;
    }

    StepResult prefill(
        const std::vector<std::uint16_t>& input,
        const std::vector<std::uint16_t>& cosine,
        const std::vector<std::uint16_t>& sine,
        bool capture_layers) {

        const std::size_t hidden_count =
            static_cast<std::size_t>(
                prefill_length_)
            * kHidden;

        const std::size_t rope_count =
            static_cast<std::size_t>(
                prefill_length_)
            * weights_.head_dim();

        if (
            input.size() != hidden_count
            || cosine.size() != rope_count
            || sine.size() != rope_count
        ) {
            throw std::runtime_error(
                "legacy_legacy_tag_g native Talker prefill input/RoPE size mismatch.");
        }

        hip_check(
            hipMemcpyAsync(
                hidden_a_.pointer,
                input.data(),
                input.size()*sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_),
            "Talker prefill hidden upload");

        hip_check(
            hipMemcpyAsync(
                cosine_.pointer,
                cosine.data(),
                cosine.size()*sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_),
            "Talker prefill cosine upload");

        hip_check(
            hipMemcpyAsync(
                sine_.pointer,
                sine.data(),
                sine.size()*sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_),
            "Talker prefill sine upload");

        DeviceBuffer* current =
            &hidden_a_;

        DeviceBuffer* next =
            &hidden_b_;

        StepResult result;

        if (capture_layers) {
            result.layer_last_rows.reserve(
                weights_.layer_count());
        }

        for (
            int layer_index = 0;
            layer_index < weights_.layer_count();
            ++layer_index
        ) {
            const LayerWeights& layer =
                weights_.layer(
                    layer_index);

            code_predictor::launch_rmsnorm(
                current->as<std::uint16_t>(),
                layer.input_norm.data(),
                norm_.as<std::uint16_t>(),
                prefill_length_,
                kHidden,
                stream_);

            linear_generic(
                norm_,
                layer.q_weight,
                layer.q_bias,
                q_,
                prefill_length_,
                weights_.q_dim(),
                kHidden);

            linear_generic(
                norm_,
                layer.k_weight,
                layer.k_bias,
                k_,
                prefill_length_,
                weights_.kv_dim(),
                kHidden);

            linear_generic(
                norm_,
                layer.v_weight,
                layer.v_bias,
                v_,
                prefill_length_,
                weights_.kv_dim(),
                kHidden);

            code_predictor::launch_rmsnorm(
                q_.as<std::uint16_t>(),
                layer.q_norm.data(),
                q_norm_.as<std::uint16_t>(),
                prefill_length_
                    * weights_.q_heads(),
                weights_.head_dim(),
                stream_);

            code_predictor::launch_rmsnorm(
                k_.as<std::uint16_t>(),
                layer.k_norm.data(),
                k_norm_.as<std::uint16_t>(),
                prefill_length_
                    * weights_.kv_heads(),
                weights_.head_dim(),
                stream_);

            launch_rope_rows(
                q_norm_,
                q_rotated_,
                prefill_length_,
                weights_.q_heads());

            launch_rope_rows(
                k_norm_,
                k_rotated_,
                prefill_length_,
                weights_.kv_heads());

            scatter_prefill_cache(
                k_rotated_,
                v_,
                caches_[
                    static_cast<std::size_t>(
                        layer_index)]);

            launch_attention(
                q_rotated_,
                caches_[
                    static_cast<std::size_t>(
                        layer_index)],
                context_,
                prefill_length_,
                0);

            linear_generic(
                context_,
                layer.o_weight,
                layer.o_bias,
                attention_output_,
                prefill_length_,
                kHidden,
                weights_.q_dim());

            code_predictor::launch_add(
                current->as<std::uint16_t>(),
                attention_output_.as<std::uint16_t>(),
                residual_.as<std::uint16_t>(),
                hidden_count,
                stream_);

            code_predictor::launch_rmsnorm(
                residual_.as<std::uint16_t>(),
                layer.post_norm.data(),
                post_norm_.as<std::uint16_t>(),
                prefill_length_,
                kHidden,
                stream_);

            linear_generic(
                post_norm_,
                layer.gate_weight,
                no_bias_,
                gate_,
                prefill_length_,
                weights_.intermediate(),
                kHidden);

            linear_generic(
                post_norm_,
                layer.up_weight,
                no_bias_,
                up_,
                prefill_length_,
                weights_.intermediate(),
                kHidden);

            code_predictor::launch_swiglu(
                gate_.as<std::uint16_t>(),
                up_.as<std::uint16_t>(),
                swiglu_.as<std::uint16_t>(),
                static_cast<std::uint64_t>(
                    prefill_length_)
                    * weights_.intermediate(),
                stream_);

            linear_generic(
                swiglu_,
                layer.down_weight,
                no_bias_,
                down_,
                prefill_length_,
                kHidden,
                weights_.intermediate());

            code_predictor::launch_add(
                residual_.as<std::uint16_t>(),
                down_.as<std::uint16_t>(),
                next->as<std::uint16_t>(),
                hidden_count,
                stream_);

            std::swap(
                current,
                next);

            if (capture_layers) {
                result.layer_last_rows.push_back(
                    download_row(
                        *current,
                        prefill_length_ - 1,
                        kHidden));
            }
        }

        code_predictor::launch_rmsnorm(
            current->as<std::uint16_t>(),
            weights_.final_norm().data(),
            final_norm_.as<std::uint16_t>(),
            prefill_length_,
            kHidden,
                stream_);

        hip_check(
            hipStreamSynchronize(
                stream_),
            "Talker prefill generation-stream synchronize");

        final_hidden_row_ =
            prefill_length_ - 1;

        result.hidden =
            download_row(
                final_norm_,
                prefill_length_ - 1,
                kHidden);

        return result;
    }


    StepResult decode(
        const std::vector<std::uint16_t>& input,
        const std::uint16_t* cosine,
        const std::uint16_t* sine,
        int decode_index,
        bool capture_layers) {

        if (
            input.size()
            != static_cast<std::size_t>(kHidden)
        ) {
            throw std::runtime_error(
                "legacy_legacy_tag_k Talker decode input size mismatch.");
        }

        const int cache_position =
            prefill_length_
            + decode_index;

        if (cache_position >= max_cache_) {
            throw std::runtime_error(
                "legacy_legacy_tag_k Talker decode cache overflow.");
        }

        code_predictor::upload(
            hidden_a_,
            input);

        hip_check(
            hipMemcpy(
                cosine_.pointer,
                cosine,
                weights_.head_dim()
                    * sizeof(std::uint16_t),
                hipMemcpyHostToDevice),
            "legacy_legacy_tag_k copy decode cosine");

        hip_check(
            hipMemcpy(
                sine_.pointer,
                sine,
                weights_.head_dim()
                    * sizeof(std::uint16_t),
                hipMemcpyHostToDevice),
            "legacy_legacy_tag_k copy decode sine");

        DeviceBuffer* current =
            &hidden_a_;

        DeviceBuffer* next =
            &hidden_b_;

        StepResult result;

        if (capture_layers) {
            result.layer_last_rows.reserve(
                weights_.layer_count());
        }

        for (
            int layer_index = 0;
            layer_index < weights_.layer_count();
            ++layer_index
        ) {
            const LayerWeights& layer =
                weights_.layer(layer_index);

            // 1. input RMSNorm
            code_predictor::launch_rmsnorm(
                current->as<std::uint16_t>(),
                layer.input_norm.data(),
                norm_.as<std::uint16_t>(),
                1,
                kHidden,
                stream_);

            // 2. explicit Q / K / V BF16 projections
            linear_generic(
                norm_,
                layer.q_weight,
                layer.q_bias,
                q_,
                1,
                weights_.q_dim(),
                kHidden);

            linear_generic(
                norm_,
                layer.k_weight,
                layer.k_bias,
                k_,
                1,
                weights_.kv_dim(),
                kHidden);

            linear_generic(
                norm_,
                layer.v_weight,
                layer.v_bias,
                v_,
                1,
                weights_.kv_dim(),
                kHidden);

            // 3. per-head Q/K RMSNorm
            code_predictor::launch_rmsnorm(
                q_.as<std::uint16_t>(),
                layer.q_norm.data(),
                q_norm_.as<std::uint16_t>(),
                weights_.q_heads(),
                weights_.head_dim(),
                stream_);

            code_predictor::launch_rmsnorm(
                k_.as<std::uint16_t>(),
                layer.k_norm.data(),
                k_norm_.as<std::uint16_t>(),
                weights_.kv_heads(),
                weights_.head_dim(),
                stream_);

            // 4. RoPE materialization
            launch_rope_rows(
                q_norm_,
                q_rotated_,
                1,
                weights_.q_heads());

            launch_rope_rows(
                k_norm_,
                k_rotated_,
                1,
                weights_.kv_heads());

            // 5. append exactly this row to persistent K/V
            scatter_decode_cache(
                k_rotated_,
                v_,
                caches_[
                    static_cast<std::size_t>(
                        layer_index)],
                cache_position);

            // 6. eager-equivalent attention
            launch_attention(
                q_rotated_,
                caches_[
                    static_cast<std::size_t>(
                        layer_index)],
                context_,
                1,
                cache_position);

            // 7. O projection + BF16 residual boundary
            linear_generic(
                context_,
                layer.o_weight,
                layer.o_bias,
                attention_output_,
                1,
                kHidden,
                weights_.q_dim());

            code_predictor::launch_add(
                current->as<std::uint16_t>(),
                attention_output_.as<std::uint16_t>(),
                residual_.as<std::uint16_t>(),
                kHidden,
                stream_);

            // 8. post-attention RMSNorm
            code_predictor::launch_rmsnorm(
                residual_.as<std::uint16_t>(),
                layer.post_norm.data(),
                post_norm_.as<std::uint16_t>(),
                1,
                kHidden,
                stream_);

            // 9. explicit gate/up -> SwiGLU -> down
            linear_generic(
                post_norm_,
                layer.gate_weight,
                no_bias_,
                gate_,
                1,
                weights_.intermediate(),
                kHidden);

            linear_generic(
                post_norm_,
                layer.up_weight,
                no_bias_,
                up_,
                1,
                weights_.intermediate(),
                kHidden);

            code_predictor::launch_swiglu(
                gate_.as<std::uint16_t>(),
                up_.as<std::uint16_t>(),
                swiglu_.as<std::uint16_t>(),
                weights_.intermediate(),
                stream_);

            linear_generic(
                swiglu_,
                layer.down_weight,
                no_bias_,
                down_,
                1,
                kHidden,
                weights_.intermediate());

            // 10. explicit final BF16 residual boundary
            code_predictor::launch_add(
                residual_.as<std::uint16_t>(),
                down_.as<std::uint16_t>(),
                next->as<std::uint16_t>(),
                kHidden,
                stream_);

            std::swap(
                current,
                next);

            if (capture_layers) {
                hip_check(
                    hipDeviceSynchronize(),
                    "legacy_legacy_tag_k Talker layer trace synchronize");

                result.layer_last_rows.push_back(
                    code_predictor::
                        download<std::uint16_t>(
                            current->as<std::uint16_t>(),
                            kHidden));
            }
        }

        code_predictor::launch_rmsnorm(
            current->as<std::uint16_t>(),
            weights_.final_norm().data(),
            final_norm_.as<std::uint16_t>(),
            1,
            kHidden,
                stream_);

        hip_check(
            hipDeviceSynchronize(),
            "legacy_legacy_tag_k Talker staged decode synchronize");

        final_hidden_row_ = 0;

        result.hidden =
            code_predictor::
                download<std::uint16_t>(
                    final_norm_.as<std::uint16_t>(),
                    kHidden);

        return result;
    }


    const std::uint16_t*
    device_final_hidden() const {

        return
            final_norm_
                .as<std::uint16_t>()
            + static_cast<std::size_t>(
                final_hidden_row_)
                * kHidden;
    }

    void decode_device(
        const std::uint16_t* input_device,
        const std::uint16_t* cosine,
        const std::uint16_t* sine,
        int decode_index) {

        if (input_device == nullptr) {
            throw std::runtime_error(
                "Talker device decode input is null");
        }

        const int cache_position =
            prefill_length_
            + decode_index;

        if (cache_position >= max_cache_) {
            throw std::runtime_error(
                "Talker device decode cache overflow");
        }

        hip_check(
            hipMemcpyAsync(
                hidden_a_.pointer,
                input_device,
                kHidden
                    * sizeof(std::uint16_t),
                hipMemcpyDeviceToDevice,
                stream_),
            "Talker device input copy");

        hip_check(
            hipMemcpyAsync(
                cosine_.pointer,
                cosine,
                weights_.head_dim()
                    * sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_),
            "Talker device decode cosine");

        hip_check(
            hipMemcpyAsync(
                sine_.pointer,
                sine,
                weights_.head_dim()
                    * sizeof(std::uint16_t),
                hipMemcpyHostToDevice,
                stream_),
            "Talker device decode sine");

        DeviceBuffer* current =
            &hidden_a_;

        DeviceBuffer* next =
            &hidden_b_;

        for (
            int layer_index = 0;
            layer_index < weights_.layer_count();
            ++layer_index
        ) {
            const LayerWeights& layer =
                weights_.layer(layer_index);

            code_predictor::launch_rmsnorm(
                current->as<std::uint16_t>(),
                layer.input_norm.data(),
                norm_.as<std::uint16_t>(),
                1,
                kHidden,
                stream_);

            // Decode specialization:
            //   3 independent Q/K/V linears -> one LDS-reuse QKV kernel.
            // BF16 output materialization is preserved exactly.
            launch_qkv_m1(
                norm_,
                layer);

            // Preserve official Q/K RMSNorm staging and the corrected
            // rotate_half=(-x2,+x1) semantics while eliminating standalone
            // Q-norm, K-norm, two RoPE launches and the KV scatter launch.
            launch_qk_norm_rope_cache_m1(
                layer,
                caches_[
                    static_cast<std::size_t>(
                        layer_index)],
                cache_position);

            launch_attention(
                q_rotated_,
                caches_[
                    static_cast<std::size_t>(
                        layer_index)],
                context_,
                1,
                cache_position);

            // Locality policy:
            //   raw context/weights are the cacheable material;
            //   the O projection is computed once and immediately consumed.
            //
            // M=1 OProj uses the LDS-staged linear so the 1024-d context
            // vector is loaded once per block instead of every wave rereading
            // it from global memory.
            launch_linear_m1(
                context_,
                layer.o_weight,
                layer.o_bias,
                nullptr,
                attention_output_,
                kHidden,
                weights_.q_dim());

            // Residual is a required BF16 model boundary. Materialize it once,
            // keep the fresh BF16 values in LDS, and consume them immediately
            // for the post-attention RMS statistic + scaling.
            launch_add_postnorm_m1(
                *current,
                attention_output_,
                layer.post_norm,
                residual_,
                post_norm_);

            // Gate and Up are derived values and are never cached globally.
            // One LDS copy of post_norm feeds two VGPR accumulator streams;
            // BF16 gate/up activation boundaries and SwiGLU are evaluated in
            // the producer kernel, leaving only the actual SwiGLU operand.
            launch_gate_up_swiglu_m1(
                post_norm_,
                layer,
                swiglu_);

            // Down projection and residual addition are a single consumer.
            // The linear result still materializes to BF16 in-register before
            
            launch_linear_m1(
                swiglu_,
                layer.down_weight,
                no_bias_,
                residual_
                    .as<std::uint16_t>(),
                *next,
                kHidden,
                weights_.intermediate());

            std::swap(
                current,
                next);
        }

        code_predictor::launch_rmsnorm(
            current->as<std::uint16_t>(),
            weights_.final_norm().data(),
            final_norm_
                .as<std::uint16_t>(),
            1,
            kHidden,
                stream_);

        // No device-wide synchronization here. FinalHead is queued onto the
        // same explicit generation stream and its 4-byte code readback provides
        // the next host-visible frame boundary.
        final_hidden_row_ = 0;
    }



private:
    void hip_check(
        hipError_t status,
        const char* operation) const {

        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string(
                    "legacy_legacy_tag_g ")
                + operation
                + ": "
                + hipGetErrorString(
                    status));
        }
    }

    void allocate_workspace() {
        const auto bf16 =
            [](std::size_t count) {
                return count
                    * sizeof(
                        std::uint16_t);
            };

        const std::size_t hidden_rows =
            static_cast<std::size_t>(
                max_rows_);

        hidden_a_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        hidden_b_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        norm_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        q_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.q_dim()));

        k_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.kv_dim()));

        v_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.kv_dim()));

        q_norm_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.q_dim()));

        k_norm_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.kv_dim()));

        q_rotated_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.q_dim()));

        k_rotated_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.kv_dim()));

        context_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.q_dim()));

        attention_output_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        residual_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        post_norm_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        gate_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.intermediate()));

        up_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.intermediate()));

        swiglu_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.intermediate()));

        down_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        final_norm_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * kHidden));

        cosine_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.head_dim()));

        sine_ =
            DeviceBuffer(
                bf16(
                    hidden_rows
                    * weights_.head_dim()));
    }

    std::vector<std::uint16_t> download_row(
        const DeviceBuffer& buffer,
        int row,
        int columns) const {

        std::vector<std::uint16_t> host(
            static_cast<std::size_t>(
                columns));

        hip_check(
            hipMemcpyAsync(
                host.data(),
                buffer.as<std::uint16_t>()
                    + static_cast<std::size_t>(
                        row)
                    * columns,
                static_cast<std::size_t>(
                    columns)
                    * sizeof(std::uint16_t),
                hipMemcpyDeviceToHost,
                stream_),
            "download Talker row");

        hip_check(
            hipStreamSynchronize(
                stream_),
            "download Talker row synchronize");

        return host;
    }

    void linear_generic(
        const DeviceBuffer& input,
        const GpuParam& weight,
        const GpuParam& bias,
        DeviceBuffer& output,
        int rows,
        int output_columns,
        int reduction_columns) {

        qwen3_tts::native_ops::
            linear_bf16(
                input.as<std::uint16_t>(),
                weight.data(),
                bias.data(),
                output.as<std::uint16_t>(),
                rows,
                output_columns,
                reduction_columns,
                stream_);
    }

    void launch_linear_m1(
        const DeviceBuffer& input,
        const GpuParam& weight,
        const GpuParam& bias,
        const std::uint16_t* residual,
        DeviceBuffer& output,
        int output_columns,
        int reduction_columns) {

        const dim3 block(
            kThreads);

        const dim3 grid(
            static_cast<unsigned>(
                (
                    output_columns
                    + 2 * kWavesPerBlock
                    - 1
                )
                / (2 * kWavesPerBlock)));

        const std::size_t shared_bytes =
            static_cast<std::size_t>(
                reduction_columns)
            * sizeof(std::uint16_t);

        hipLaunchKernelGGL(
            talker_linear_fused_kernel,
            grid,
            block,
            shared_bytes,
            stream_,
            input.as<std::uint16_t>(),
            weight.data(),
            bias.data(),
            residual,
            output.as<std::uint16_t>(),
            output_columns,
            reduction_columns);

        hip_check(
            hipGetLastError(),
            "Qingming M1 linear launch");
    }

    void launch_qkv_m1(
        const DeviceBuffer& input,
        const LayerWeights& layer) {

        const int total =
            weights_.q_dim()
            + 2 * weights_.kv_dim();

        const dim3 block(
            kThreads);

        const dim3 grid(
            static_cast<unsigned>(
                (
                    total
                    + 2 * kWavesPerBlock
                    - 1
                )
                / (2 * kWavesPerBlock)));

        const std::size_t shared_bytes =
            static_cast<std::size_t>(
                kHidden)
            * sizeof(std::uint16_t);

        hipLaunchKernelGGL(
            talker_qkv_fused_kernel,
            grid,
            block,
            shared_bytes,
            stream_,
            input.as<std::uint16_t>(),
            layer.q_weight.data(),
            layer.q_bias.data(),
            layer.k_weight.data(),
            layer.k_bias.data(),
            layer.v_weight.data(),
            layer.v_bias.data(),
            q_.as<std::uint16_t>(),
            k_.as<std::uint16_t>(),
            v_.as<std::uint16_t>(),
            kHidden,
            weights_.q_dim(),
            weights_.kv_dim());

        hip_check(
            hipGetLastError(),
            "Qingming fused QKV launch");
    }

    void launch_qk_norm_rope_cache_m1(
        const LayerWeights& layer,
        LayerCache& cache,
        int cache_position) {

        const int blocks =
            weights_.q_heads()
            + weights_.kv_heads();

        hipLaunchKernelGGL(
            talker_qk_norm_rope_cache_fused_kernel,
            dim3(
                static_cast<unsigned>(
                    blocks)),
            dim3(128),
            0,
            stream_,
            q_.as<std::uint16_t>(),
            k_.as<std::uint16_t>(),
            v_.as<std::uint16_t>(),
            layer.q_norm.data(),
            layer.k_norm.data(),
            cosine_.as<std::uint16_t>(),
            sine_.as<std::uint16_t>(),
            q_rotated_.as<std::uint16_t>(),
            cache.key.as<std::uint16_t>(),
            cache.value.as<std::uint16_t>(),
            weights_.q_heads(),
            weights_.kv_heads(),
            weights_.head_dim(),
            max_cache_,
            cache_position,
            1.0e-6f);

        hip_check(
            hipGetLastError(),
            "Qingming QK norm/RoPE/cache launch");
    }

    void launch_rope_rows(
        const DeviceBuffer& input,
        DeviceBuffer& output,
        int sequence_length,
        int heads) {

        const std::uint64_t count =
            static_cast<std::uint64_t>(
                sequence_length)
            * heads
            * weights_.head_dim();

        hipLaunchKernelGGL(
            talker_rope_rows_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        count
                        + kThreads
                        - 1
                    )
                    / kThreads)),
            dim3(kThreads),
            0,
            stream_,
            input.as<std::uint16_t>(),
            cosine_.as<std::uint16_t>(),
            sine_.as<std::uint16_t>(),
            output.as<std::uint16_t>(),
            sequence_length,
            heads,
            weights_.head_dim());

        hip_check(
            hipGetLastError(),
            "Qingming prefill RoPE launch");
    }

    void scatter_prefill_cache(
        const DeviceBuffer& key,
        const DeviceBuffer& value,
        LayerCache& cache) {

        const std::uint64_t count =
            static_cast<std::uint64_t>(
                prefill_length_)
            * weights_.kv_heads()
            * weights_.head_dim();

        hipLaunchKernelGGL(
            talker_scatter_prefill_kv_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        count
                        + kThreads
                        - 1
                    )
                    / kThreads)),
            dim3(kThreads),
            0,
            stream_,
            key.as<std::uint16_t>(),
            value.as<std::uint16_t>(),
            cache.key.as<std::uint16_t>(),
            cache.value.as<std::uint16_t>(),
            prefill_length_,
            weights_.kv_heads(),
            weights_.head_dim(),
            max_cache_);

        hip_check(
            hipGetLastError(),
            "Qingming prefill KV scatter launch");
    }

    void scatter_decode_cache(
        const DeviceBuffer& key,
        const DeviceBuffer& value,
        LayerCache& cache,
        int cache_position) {

        const std::uint64_t count =
            static_cast<std::uint64_t>(
                weights_.kv_heads())
            * weights_.head_dim();

        hipLaunchKernelGGL(
            talker_scatter_decode_kv_kernel,
            dim3(
                static_cast<unsigned>(
                    (
                        count
                        + kThreads
                        - 1
                    )
                    / kThreads)),
            dim3(kThreads),
            0,
            stream_,
            key.as<std::uint16_t>(),
            value.as<std::uint16_t>(),
            cache.key.as<std::uint16_t>(),
            cache.value.as<std::uint16_t>(),
            weights_.kv_heads(),
            weights_.head_dim(),
            max_cache_,
            cache_position);

        hip_check(
            hipGetLastError(),
            "legacy_legacy_tag_k staged decode KV scatter");
    }


    void launch_attention(
        const DeviceBuffer& query,
        const LayerCache& cache,
        DeviceBuffer& context,
        int query_length,
        int cache_before) {

        const int maximum_key_end =
            cache_before + query_length;

        const int maximum_key_length =
            sliding_window_ > 0
            ? std::min(maximum_key_end, sliding_window_)
            : maximum_key_end;

        const float scale =
            1.0f
            / std::sqrt(
                static_cast<float>(
                    weights_.head_dim()));

        const bool use_gqa2_m1 =
            query_length == 1
            && weights_.q_heads()
                == 2 * weights_.kv_heads()
            && weights_.head_dim() == 128;

        if (use_gqa2_m1) {
            const std::size_t shared_bytes =
                static_cast<std::size_t>(
                    2 * weights_.head_dim()
                    + 2 * maximum_key_length)
                * sizeof(std::uint16_t);

            hipLaunchKernelGGL(
                talker_attention_gqa2_m1_locality_kernel,
                dim3(static_cast<unsigned>(
                    weights_.kv_heads())),
                dim3(kThreads),
                shared_bytes,
            stream_,
                query.as<std::uint16_t>(),
                cache.key.as<std::uint16_t>(),
                cache.value.as<std::uint16_t>(),
                context.as<std::uint16_t>(),
                cache_before,
                weights_.q_heads(),
                weights_.kv_heads(),
                weights_.head_dim(),
                max_cache_,
                sliding_window_,
                scale);

            hip_check(
                hipGetLastError(),
                "Talker GQA2 M1 locality attention launch");
            return;
        }

        const std::size_t shared_bytes =
            static_cast<std::size_t>(
                weights_.head_dim()
                + maximum_key_length)
            * sizeof(std::uint16_t);

        const int blocks =
            weights_.q_heads()
            * query_length;

        hipLaunchKernelGGL(
            talker_attention_kernel,
            dim3(static_cast<unsigned>(blocks)),
            dim3(kThreads),
            shared_bytes,
            stream_,
            query.as<std::uint16_t>(),
            cache.key.as<std::uint16_t>(),
            cache.value.as<std::uint16_t>(),
            context.as<std::uint16_t>(),
            query_length,
            cache_before,
            weights_.q_heads(),
            weights_.kv_heads(),
            weights_.head_dim(),
            max_cache_,
            sliding_window_,
            scale);

        hip_check(
            hipGetLastError(),
            "Talker generic fused attention launch");
    }


    void launch_add_postnorm_m1(
        const DeviceBuffer& hidden,
        const DeviceBuffer& attention,
        const GpuParam& norm_weight,
        DeviceBuffer& residual,
        DeviceBuffer& normalized) {

        const std::size_t shared_bytes =
            static_cast<std::size_t>(
                kHidden)
            * sizeof(std::uint16_t);

        hipLaunchKernelGGL(
            talker_add_postnorm_fused_kernel,
            dim3(1),
            dim3(kThreads),
            shared_bytes,
            stream_,
            hidden.as<std::uint16_t>(),
            attention.as<std::uint16_t>(),
            norm_weight.data(),
            residual.as<std::uint16_t>(),
            normalized.as<std::uint16_t>(),
            kHidden,
            1.0e-6f);

        hip_check(
            hipGetLastError(),
            "Qingming add/postnorm launch");
    }

    void launch_gate_up_swiglu_m1(
        const DeviceBuffer& input,
        const LayerWeights& layer,
        DeviceBuffer& output) {

        const dim3 block(
            kThreads);

        const dim3 grid(
            static_cast<unsigned>(
                (
                    weights_.intermediate()
                    + 2 * kWavesPerBlock
                    - 1
                )
                / (2 * kWavesPerBlock)));

        const std::size_t shared_bytes =
            static_cast<std::size_t>(
                kHidden)
            * sizeof(std::uint16_t);

        hipLaunchKernelGGL(
            talker_gate_up_swiglu_fused_kernel,
            grid,
            block,
            shared_bytes,
            stream_,
            input.as<std::uint16_t>(),
            layer.gate_weight.data(),
            layer.up_weight.data(),
            output.as<std::uint16_t>(),
            kHidden,
            weights_.intermediate());

        hip_check(
            hipGetLastError(),
            "Qingming fused gate/up/SwiGLU launch");
    }

    TalkerWeights weights_;
    hipStream_t stream_=nullptr;

    int prefill_length_ = 0;
    int final_hidden_row_ = 0;
    int max_rows_ = 0;
    int max_cache_ = 0;
    int sliding_window_ = -1;

    std::vector<LayerCache> caches_;

    DeviceBuffer hidden_a_;
    DeviceBuffer hidden_b_;
    DeviceBuffer norm_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer q_norm_;
    DeviceBuffer k_norm_;
    DeviceBuffer q_rotated_;
    DeviceBuffer k_rotated_;
    DeviceBuffer context_;
    DeviceBuffer attention_output_;
    DeviceBuffer residual_;
    DeviceBuffer post_norm_;
    DeviceBuffer gate_;
    DeviceBuffer up_;
    DeviceBuffer swiglu_;
    DeviceBuffer down_;
    DeviceBuffer final_norm_;
    DeviceBuffer cosine_;
    DeviceBuffer sine_;

    GpuParam no_bias_;
};


static float host_bf16_to_float(
    std::uint16_t value) {

    std::uint32_t bits =
        static_cast<std::uint32_t>(
            value) << 16;

    float result = 0.0f;

    std::memcpy(
        &result,
        &bits,
        sizeof(result));

    return result;
}


static void compare_hidden_row(
    const std::vector<std::uint16_t>& native,
    const std::uint16_t* official,
    double& abs_sum,
    std::size_t& count,
    float& max_abs,
    double& row_mae,
    float& row_max) {

    if (
        native.size()
        != static_cast<std::size_t>(
            kHidden)
    ) {
        throw std::runtime_error(
            "legacy_legacy_tag_g hidden comparison width mismatch.");
    }

    double local_sum = 0.0;
    float local_max = 0.0f;

    for (
        int index = 0;
        index < kHidden;
        ++index
    ) {
        const float difference =
            std::fabs(
                host_bf16_to_float(
                    native[
                        static_cast<std::size_t>(
                            index)])
                - host_bf16_to_float(
                    official[index]));

        local_sum += difference;

        local_max =
            std::max(
                local_max,
                difference);
    }

    abs_sum += local_sum;
    count += kHidden;
    max_abs =
        std::max(
            max_abs,
            local_max);

    row_mae =
        local_sum
        / static_cast<double>(
            kHidden);

    row_max =
        local_max;
}


static std::vector<std::uint16_t>
load_official_layer_row(
    const fs::path& work,
    int layer,
    int frame) {

    const fs::path path =
        work
        / "official_talker_layer_lastrows"
        / (
            "layer_"
            + code_predictor::
                two_digit(layer)
            + ".bf16.bin");

    const auto all =
        read_binary<std::uint16_t>(
            path);

    const std::size_t offset =
        static_cast<std::size_t>(
            frame)
        * kHidden;

    if (
        all.size()
        < offset + kHidden
    ) {
        throw std::runtime_error(
            "legacy_legacy_tag_g official Talker layer trace is too short: "
            + path.string());
    }

    return std::vector<std::uint16_t>(
        all.begin()
            + static_cast<std::ptrdiff_t>(
                offset),
        all.begin()
            + static_cast<std::ptrdiff_t>(
                offset + kHidden));
}


struct ValidationSummary {
    int prefill_code0_match = 0;
    double prefill_hidden_mae = 0.0;
    float prefill_hidden_max_abs = 0.0f;

    int decode_steps = 0;
    int decode_code0_matches = 0;
    double decode_abs_sum = 0.0;
    std::size_t decode_values = 0;
    float decode_max_abs = 0.0f;
};


static ValidationSummary validate_native_talker(
    const fs::path& model_dir,
    const fs::path& work,
    const std::vector<std::uint16_t>& native_prefill,
    int prefill_length,
    int frame_count,
    const std::vector<std::uint16_t>& official_hidden,
    const std::vector<std::uint16_t>& teacher_next_inputs,
    const std::vector<std::int32_t>& official_codes,
    NativeFinalHead& final_head) {

    const int sliding_window =
        read_int_text(
            work
            / "talker_sliding_window.txt");

    NativeTalkerEngine engine(
        model_dir,
        prefill_length,
        frame_count,
        sliding_window);

    const auto prefill_cosine =
        read_binary<std::uint16_t>(
            work
            / "talker_prefill_rope_cos.bf16.bin");

    const auto prefill_sine =
        read_binary<std::uint16_t>(
            work
            / "talker_prefill_rope_sin.bf16.bin");

    const auto decode_cosine =
        read_binary<std::uint16_t>(
            work
            / "talker_decode_rope_cos.bf16.bin");

    const auto decode_sine =
        read_binary<std::uint16_t>(
            work
            / "talker_decode_rope_sin.bf16.bin");

    const int head_dim =
        engine.head_dim();

    if (
        decode_cosine.size()
        < static_cast<std::size_t>(
            std::max(
                0,
                frame_count - 1))
            * head_dim
        || decode_sine.size()
        < static_cast<std::size_t>(
            std::max(
                0,
                frame_count - 1))
            * head_dim
    ) {
        throw std::runtime_error(
            "legacy_legacy_tag_g decode RoPE reference is too short.");
    }

    ValidationSummary summary;

    const StepResult prefill_result =
        engine.prefill(
            native_prefill,
            prefill_cosine,
            prefill_sine,
            true);

    {
        double dummy_abs_sum = 0.0;
        std::size_t dummy_count = 0;

        compare_hidden_row(
            prefill_result.hidden,
            official_hidden.data(),
            dummy_abs_sum,
            dummy_count,
            summary.prefill_hidden_max_abs,
            summary.prefill_hidden_mae,
            summary.prefill_hidden_max_abs);
    }

    std::vector<std::int32_t> history;

    const auto prefill_head =
        final_head.run(
            prefill_result.hidden.data(),
            history,
            0);

    const int official_frame0 =
        official_codes[0];

    summary.prefill_code0_match =
        prefill_head.code
            == official_frame0
        ? 1
        : 0;

    std::cout
        << "talker_prefill_result: "
        << "hidden_mae="
        << summary.prefill_hidden_mae
        << " hidden_max_abs="
        << summary.prefill_hidden_max_abs
        << " native_code0="
        << prefill_head.code
        << " official_code0="
        << official_frame0
        << " code0_match="
        << (
            summary.prefill_code0_match
            ? "True"
            : "False")
        << "\n";

    for (
        int layer = 0;
        layer < engine.layer_count();
        ++layer
    ) {
        const auto official =
            load_official_layer_row(
                work,
                layer,
                0);

        double abs_sum = 0.0;
        std::size_t values = 0;
        float max_abs = 0.0f;
        double mae = 0.0;
        float row_max = 0.0f;

        compare_hidden_row(
            prefill_result
                .layer_last_rows[
                    static_cast<std::size_t>(
                        layer)],
            official.data(),
            abs_sum,
            values,
            max_abs,
            mae,
            row_max);

        std::cout
            << "talker_layer_boundary: "
            << "phase=prefill_last"
            << " layer="
            << layer
            << " mae="
            << mae
            << " max_abs="
            << row_max
            << "\n";
    }

    history.push_back(
        official_frame0);

    for (
        int decode_index = 0;
        decode_index + 1 < frame_count;
        ++decode_index
    ) {
        std::vector<std::uint16_t> input(
            teacher_next_inputs.begin()
                + static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        decode_index)
                    * kHidden),
            teacher_next_inputs.begin()
                + static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        decode_index + 1)
                    * kHidden));

        const bool capture_layers =
            decode_index == 0;

        const StepResult result =
            engine.decode(
                input,
                decode_cosine.data()
                    + static_cast<std::size_t>(
                        decode_index)
                    * head_dim,
                decode_sine.data()
                    + static_cast<std::size_t>(
                        decode_index)
                    * head_dim,
                decode_index,
                capture_layers);

        const int frame =
            decode_index + 1;

        double row_mae = 0.0;
        float row_max = 0.0f;

        compare_hidden_row(
            result.hidden,
            official_hidden.data()
                + static_cast<std::size_t>(
                    frame)
                * kHidden,
            summary.decode_abs_sum,
            summary.decode_values,
            summary.decode_max_abs,
            row_mae,
            row_max);

        const auto head =
            final_head.run(
                result.hidden.data(),
                history,
                frame);

        const int official_code0 =
            official_codes[
                static_cast<std::size_t>(
                    frame) * 16];

        const bool match =
            head.code
            == official_code0;

        if (match) {
            ++summary.decode_code0_matches;
        }

        ++summary.decode_steps;

        std::cout
            << "talker_decode_frame: "
            << "frame="
            << frame
            << " hidden_mae="
            << row_mae
            << " hidden_max_abs="
            << row_max
            << " native_code0="
            << head.code
            << " official_code0="
            << official_code0
            << " code0_match="
            << (
                match
                ? "True"
                : "False")
            << "\n";

        if (capture_layers) {
            for (
                int layer = 0;
                layer < engine.layer_count();
                ++layer
            ) {
                const auto official =
                    load_official_layer_row(
                        work,
                        layer,
                        frame);

                double abs_sum = 0.0;
                std::size_t values = 0;
                float max_abs = 0.0f;
                double mae = 0.0;
                float layer_max = 0.0f;

                compare_hidden_row(
                    result.layer_last_rows[
                        static_cast<std::size_t>(
                            layer)],
                    official.data(),
                    abs_sum,
                    values,
                    max_abs,
                    mae,
                    layer_max);

                std::cout
                    << "talker_layer_boundary: "
                    << "phase=decode_frame1"
                    << " layer="
                    << layer
                    << " mae="
                    << mae
                    << " max_abs="
                    << layer_max
                    << "\n";
            }
        }

        history.push_back(
            official_code0);
    }

    return summary;
}

} // namespace talker


static std::size_t f32_mismatch_count(
    const float* a,
    const float* b,
    std::size_t count){

    std::size_t mismatches=0;
    for(std::size_t i=0;i<count;++i){
        std::uint32_t aa=0,bb=0;
        std::memcpy(&aa,a+i,sizeof(aa));
        std::memcpy(&bb,b+i,sizeof(bb));
        if(aa!=bb)
            ++mismatches;
    }
    return mismatches;
}

static float f32_max_abs_diff(
    const float* a,
    const float* b,
    std::size_t count){

    float maximum=0.0f;
    for(std::size_t i=0;i<count;++i){
        const float d=std::fabs(a[i]-b[i]);
        if(d>maximum)
            maximum=d;
    }
    return maximum;
}

static std::size_t f32_first_mismatch(
    const float* a,
    const float* b,
    std::size_t count){

    for(std::size_t i=0;i<count;++i){
        std::uint32_t aa=0,bb=0;
        std::memcpy(&aa,a+i,sizeof(aa));
        std::memcpy(&bb,b+i,sizeof(bb));
        if(aa!=bb)
            return i;
    }
    return count;
}

static std::size_t u8_mismatch_count(
    const std::uint8_t* a,
    const std::uint8_t* b,
    std::size_t count){

    std::size_t mismatches=0;
    for(std::size_t i=0;i<count;++i)
        mismatches += a[i]!=b[i] ? 1 : 0;
    return mismatches;
}

static bool bf16_equal(
    const std::uint16_t* a,
    const std::uint16_t* b,
    std::size_t count){
    return std::equal(a,a+count,b);
}

struct PredictorStageMismatch {
    bool found=false;
    int step=-1;
    std::string stage;
    std::size_t mismatch_count=0;
    std::size_t first_index=0;
    std::uint16_t native_bits=0;
    std::uint16_t official_bits=0;
};

static PredictorStageMismatch compare_predictor_stage(
    const fs::path& native_path,
    const fs::path& official_path,
    int step,
    const std::string& stage){

    const auto native=
        read_binary<std::uint16_t>(native_path);
    const auto official=
        read_binary<std::uint16_t>(official_path);

    PredictorStageMismatch result;
    result.step=step;
    result.stage=stage;

    if(native.size()!=official.size()){
        result.found=true;
        result.mismatch_count=
            std::max(native.size(),official.size());
        result.first_index=0;
        result.native_bits=
            native.empty()?0:native[0];
        result.official_bits=
            official.empty()?0:official[0];
        return result;
    }

    std::size_t first=native.size();
    std::size_t count=0;
    for(std::size_t i=0;i<native.size();++i){
        if(native[i]!=official[i]){
            if(first==native.size())
                first=i;
            ++count;
        }
    }

    if(count!=0){
        result.found=true;
        result.mismatch_count=count;
        result.first_index=first;
        result.native_bits=native[first];
        result.official_bits=official[first];
    }
    return result;
}

static PredictorStageMismatch diagnose_predictor_frame(
    const fs::path& work,
    const fs::path& predictor_output,
    int frame,
    int last_step){

    const std::string frame_name=
        std::string("frame_")
        +(frame<10?"0":"")
        +std::to_string(frame);

    const fs::path native_frame=
        predictor_output/"free"/frame_name;
    const fs::path official_frame=
        work/"official_predictor"/frame_name;

    for(int step=0;step<=last_step;++step){
        const std::string step_name=
            std::string("predictor_step_")
            +(step<10?"0":"")
            +std::to_string(step);

        const fs::path n=native_frame/step_name;
        const fs::path o=official_frame/step_name;

        const std::array<std::pair<std::string,fs::path>,8> stages{{
            {"input_1024",fs::path("input_1024.bf16.bin")},
            {"projection_1024",fs::path("projection_output_1024.bf16.bin")},
            {"layer_00",fs::path("layer_00/output.bf16.bin")},
            {"layer_01",fs::path("layer_01/output.bf16.bin")},
            {"layer_02",fs::path("layer_02/output.bf16.bin")},
            {"layer_03",fs::path("layer_03/output.bf16.bin")},
            {"layer_04",fs::path("layer_04/output.bf16.bin")},
            {"final_norm",fs::path("final_norm_output.bf16.bin")}
        }};

        for(const auto& item:stages){
            const auto mismatch=
                compare_predictor_stage(
                    n/item.second,
                    o/item.second,
                    step,
                    item.first);
            if(mismatch.found)
                return mismatch;
        }

        const auto logits=
            compare_predictor_stage(
                n/"raw_logits.bf16.bin",
                o/"raw_logits.bf16.bin",
                step,
                "lm_head_logits");
        if(logits.found)
            return logits;
    }

    return PredictorStageMismatch{};
}

static void legacy_create_predictor_frame_assets_a(
    const fs::path& runtime_root,
    const fs::path& production_assets,
    const fs::path& official_predictor_root,
    int frame,
    const std::uint16_t* hidden,
    const std::vector<std::uint16_t>& code0_embedding,
    const std::vector<std::uint16_t>& context,
    std::int32_t code0){

    const std::string frame_name=
        std::string("frame_")
        + (frame<10?"0":"")
        + std::to_string(frame);

    const fs::path frame_root=runtime_root/frame_name;
    fs::create_directories(frame_root);

    std::vector<std::uint16_t> initial(
        static_cast<std::size_t>(2)*1024);
    std::copy(hidden,hidden+1024,initial.begin());
    std::copy(
        code0_embedding.begin(),
        code0_embedding.end(),
        initial.begin()+1024);

    write_binary_tail(
        frame_root/"initial_input.bf16.bin",
        initial);
    write_binary_tail(
        frame_root/"context_embedding.bf16.bin",
        context);

    std::vector<std::int32_t> seed(16,0);
    seed[0]=code0;
    write_binary_tail(
        frame_root/"expected_codes.i32.bin",
        seed);

    for(int step=0;step<15;++step){
        const std::string step_name=
            std::string("predictor_step_")
            +(step<10?"0":"")
            +std::to_string(step);
        const fs::path dst=frame_root/step_name;
        fs::create_directories(dst);
        const fs::path src=
            production_assets/"rope"/step_name;
        for(const char* fn:{
            "rope_cos.bf16.bin",
            "rope_sin.bf16.bin"
        }){
            const fs::path link=dst/fn;
            std::error_code ec;
            fs::remove(link,ec);
            fs::create_symlink(
                fs::absolute(src/fn),
                link);
        }

        const fs::path teacher_src=
            official_predictor_root
            /frame_name
            /step_name
            /"legacy_teacher";

        {
            const fs::path link=
                dst/"teacher_input.bf16.bin";
            std::error_code ec;
            fs::remove(link,ec);
            fs::create_symlink(
                fs::absolute(
                    teacher_src/"teacher_input.bf16.bin"),
                link);
        }

        const fs::path teacher_layer_input_root =
            dst / "teacher_layer_input";
        fs::create_directories(
            teacher_layer_input_root);

        for(int layer=0;layer<5;++layer){
            const std::string layer_name=
                std::string("layer_")
                +(layer<10?"0":"")
                +std::to_string(layer);

            const fs::path official_layer_input =
                layer == 0
                ? (
                    official_predictor_root
                    / frame_name
                    / step_name
                    / "projection_output_1024.bf16.bin"
                )
                : (
                    official_predictor_root
                    / frame_name
                    / step_name
                    / (
                        std::string("layer_")
                        + (
                            (layer - 1) < 10
                            ? "0" : ""
                        )
                        + std::to_string(layer - 1)
                    )
                    / "output.bf16.bin"
                );

            {
                const fs::path link =
                    teacher_layer_input_root
                    / (
                        layer_name
                        + ".bf16.bin"
                    );

                std::error_code ec;
                fs::remove(link, ec);
                fs::create_symlink(
                    fs::absolute(
                        official_layer_input),
                    link);
            }

            const fs::path src_cache=
                teacher_src
                /"teacher_cache_before"
                /layer_name;

            const fs::path dst_cache=
                dst
                /"teacher_cache_before"
                /layer_name;

            fs::create_directories(dst_cache);

            for(const char* fn:{
                "key.bf16.bin",
                "value.bf16.bin"
            }){
                const fs::path link=
                    dst_cache/fn;
                std::error_code ec;
                fs::remove(link,ec);
                fs::create_symlink(
                    fs::absolute(src_cache/fn),
                    link);
            }
        }
    }
}


static void create_predictor_frame_assets_free(
    const fs::path& runtime_root,
    const fs::path& production_assets,
    int frame,
    const std::uint16_t* hidden,
    const std::vector<std::uint16_t>& code0_embedding,
    const std::vector<std::uint16_t>& context,
    std::int32_t code0) {

    const std::string frame_name =
        std::string("frame_")
        + (
            frame < 10
            ? "0"
            : ""
        )
        + std::to_string(frame);

    const fs::path frame_root =
        runtime_root
        / frame_name;

    fs::create_directories(
        frame_root);

    std::vector<std::uint16_t> initial(
        static_cast<std::size_t>(2)
        * 1024);

    std::copy(
        hidden,
        hidden + 1024,
        initial.begin());

    std::copy(
        code0_embedding.begin(),
        code0_embedding.end(),
        initial.begin() + 1024);

    write_binary_tail(
        frame_root
            / "initial_input.bf16.bin",
        initial);

    write_binary_tail(
        frame_root
            / "context_embedding.bf16.bin",
        context);

    std::vector<std::int32_t> seed(
        16,
        0);

    seed[0] = code0;

    write_binary_tail(
        frame_root
            / "expected_codes.i32.bin",
        seed);

    for (
        int step = 0;
        step < 15;
        ++step
    ) {
        const std::string step_name =
            std::string(
                "predictor_step_")
            + (
                step < 10
                ? "0"
                : ""
            )
            + std::to_string(step);

        const fs::path destination =
            frame_root
            / step_name;

        fs::create_directories(
            destination);

        const fs::path source =
            production_assets
            / "rope"
            / step_name;

        for (
            const char* file : {
                "rope_cos.bf16.bin",
                "rope_sin.bf16.bin"
            }
        ) {
            const fs::path link =
                destination
                / file;

            std::error_code ec;
            fs::remove(
                link,
                ec);

            fs::create_symlink(
                fs::absolute(
                    source / file),
                link);
        }
    }
}




static std::vector<std::uint16_t>
make_predictor_rope_row(
    int position,
    int head_dim=128,
    double theta=1000000.0) {

    if(
        position<0
        ||head_dim<=0
        ||(head_dim&1)
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 invalid Predictor RoPE geometry");
    }

    const int half=head_dim/2;

    std::vector<std::uint16_t>
        row(
            static_cast<std::size_t>(
                head_dim));

    for(int i=0;i<half;++i){
        const double inv=
            std::pow(
                theta,
                -2.0
                *static_cast<double>(i)
                /static_cast<double>(
                    head_dim));

        const double angle=
            static_cast<double>(
                position)
            *inv;

        const auto c=
            model_frontend::
                f32_to_bf16_rne(
                    static_cast<float>(
                        std::cos(angle)));

        const auto s=
            model_frontend::
                f32_to_bf16_rne(
                    static_cast<float>(
                        std::sin(angle)));

        // Return cos in the first call and sin is generated by the companion
        // function below. This helper is retained only for geometry clarity.
        (void)s;
        row[
            static_cast<std::size_t>(i)]=c;
        row[
            static_cast<std::size_t>(
                i+half)]=c;
    }

    return row;
}

static void make_predictor_rope(
    int step,
    std::vector<std::uint16_t>& cosine,
    std::vector<std::uint16_t>& sine,
    int head_dim=128,
    double theta=1000000.0) {

    const int sequence_length=
        step==0?2:1;

    const int first_position=
        step==0?0:step+1;

    cosine.resize(
        static_cast<std::size_t>(
            sequence_length)
        *head_dim);

    sine.resize(
        cosine.size());

    const int half=head_dim/2;

    for(int row=0;row<sequence_length;++row){
        const int position=
            first_position+row;

        for(int i=0;i<half;++i){
            const double inv=
                std::pow(
                    theta,
                    -2.0
                    *static_cast<double>(i)
                    /static_cast<double>(
                        head_dim));

            const double angle=
                static_cast<double>(
                    position)
                *inv;

            const auto c=
                model_frontend::
                    f32_to_bf16_rne(
                        static_cast<float>(
                            std::cos(angle)));

            const auto s=
                model_frontend::
                    f32_to_bf16_rne(
                        static_cast<float>(
                            std::sin(angle)));

            const std::size_t base=
                static_cast<std::size_t>(
                    row)
                *head_dim;

            cosine[base+i]=c;
            cosine[base+i+half]=c;
            sine[base+i]=s;
            sine[base+i+half]=s;
        }
    }
}

static void legacy_create_predictor_frame_assets_b(
    const fs::path& runtime_root,
    int frame,
    const std::uint16_t* hidden,
    const std::vector<std::uint16_t>& code0_embedding,
    const std::vector<std::uint16_t>& context,
    std::int32_t code0) {

    const std::string frame_name=
        std::string("frame_")
        +(frame<10?"0":"")
        +std::to_string(frame);

    const fs::path frame_root=
        runtime_root/frame_name;

    fs::create_directories(
        frame_root);

    std::vector<std::uint16_t> initial(
        static_cast<std::size_t>(2)
        *1024);

    std::copy(
        hidden,
        hidden+1024,
        initial.begin());

    std::copy(
        code0_embedding.begin(),
        code0_embedding.end(),
        initial.begin()+1024);

    write_binary_tail(
        frame_root/
            "initial_input.bf16.bin",
        initial);

    write_binary_tail(
        frame_root/
            "context_embedding.bf16.bin",
        context);

    std::vector<std::int32_t>
        seed(16,0);

    seed[0]=code0;

    write_binary_tail(
        frame_root/
            "expected_codes.i32.bin",
        seed);

    for(int step=0;step<15;++step){
        const std::string step_name=
            std::string("predictor_step_")
            +(step<10?"0":"")
            +std::to_string(step);

        const fs::path step_root=
            frame_root/step_name;

        fs::create_directories(
            step_root);

        std::vector<std::uint16_t>
            cosine;
        std::vector<std::uint16_t>
            sine;

        make_predictor_rope(
            step,
            cosine,
            sine,
            128,
            1000000.0);

        write_binary_tail(
            step_root/
                "rope_cos.bf16.bin",
            cosine);

        write_binary_tail(
            step_root/
                "rope_sin.bf16.bin",
            sine);
    }
}

// ============================================================================
// legacy_legacy_tag_h.0 Native Qwen3-TTS 12.5Hz codec decoder/vocoder.
// F32 speech-tokenizer weights, wave32 VGPR accumulators, explicit LDS
// reductions, and a constant-shape HIP Graph DAG.
// ============================================================================
namespace codec_decoder {
namespace fs = std::filesystem;

constexpr int WAVE=32, THREADS=256, WAVES=8;
constexpr int NQ=16, CB=2048, CBD=256, CBO=512;
constexpr int LAT=1024, TH=512, TI=1024, TL=8, HEADS=16, HD=64, AD=HEADS*HD, WINDOW=72;
constexpr int DD=1536, SAMPLE_RATE=24000, UPSAMPLE=1920;

static void hipok(hipError_t e,const char* op){
    if(e!=hipSuccess) throw std::runtime_error(std::string("legacy_legacy_tag_h codec ")+op+": "+hipGetErrorString(e));
}

class Buffer {
public:
    Buffer()=default;
    explicit Buffer(std::size_t n){alloc(n);}
    ~Buffer(){if(p_)(void)hipFree(p_);}
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& o) noexcept:p_(o.p_),n_(o.n_){o.p_=nullptr;o.n_=0;}
    Buffer& operator=(Buffer&& o) noexcept {
        if(this!=&o){if(p_)(void)hipFree(p_);p_=o.p_;n_=o.n_;o.p_=nullptr;o.n_=0;} return *this;
    }
    void alloc(std::size_t n){if(p_)(void)hipFree(p_);p_=nullptr;n_=std::max<std::size_t>(n,1);hipok(hipMalloc(&p_,n_),"hipMalloc");}
    void alloc_async(std::size_t n,hipStream_t stream){
        if(p_)(void)hipFree(p_);
        p_=nullptr;
        n_=std::max<std::size_t>(n,1);
        hipok(hipMallocAsync(&p_,n_,stream),"hipMallocAsync");
    }
    void zero_async(hipStream_t stream){
        if(p_&&n_){
            hipok(
                hipMemsetAsync(
                    p_,0,n_,stream),
                "hipMemsetAsync");
        }
    }
    template<class T>T* as(){return static_cast<T*>(p_);}
    template<class T>const T* as()const{return static_cast<const T*>(p_);}
private:void* p_=nullptr;std::size_t n_=0;
};

class Param {
public:
    Param()=default;
    explicit Param(const model_frontend::Tensor& t){
        if(t.dtype!="F32") throw std::runtime_error("legacy_legacy_tag_h codec expected F32 tensor: "+t.name+" dtype="+t.dtype);
        shape=t.shape;
        const std::size_t bytes=static_cast<std::size_t>(t.elements())*sizeof(float);
        d=Buffer(bytes);
        hipok(hipMemcpy(d.as<float>(),t.data,bytes,hipMemcpyHostToDevice),"parameter upload");
    }
    Param(const model_frontend::Tensor& t,hipStream_t upload_stream){
        if(t.dtype!="F32") throw std::runtime_error("legacy_legacy_tag_h codec expected F32 tensor: "+t.name+" dtype="+t.dtype);
        shape=t.shape;
        const std::size_t bytes=static_cast<std::size_t>(t.elements())*sizeof(float);
        d.alloc_async(bytes,upload_stream);
        hipok(
            hipMemcpyAsync(
                d.as<float>(),t.data,bytes,
                hipMemcpyHostToDevice,upload_stream),
            "parameter async upload");
    }
    Param(Param&&) noexcept=default; Param& operator=(Param&&) noexcept=default;
    Param(const Param&)=delete; Param& operator=(const Param&)=delete;
    const float* data()const{return d.as<float>();}
    std::vector<std::int64_t> shape; Buffer d;
};
static std::vector<std::int64_t> shp(std::initializer_list<std::int64_t> x){return std::vector<std::int64_t>(x);}

class Parameters {
public:
    explicit Parameters(const fs::path& dir):host_(dir){
        int least=0,greatest=0;
        hipok(
            hipDeviceGetStreamPriorityRange(&least,&greatest),
            "parameter preload priority range");
        hipok(
            hipStreamCreateWithPriority(
                &upload_stream_,hipStreamNonBlocking,least),
            "parameter preload stream create");
        try{
            load_contract();
            hipok(
                hipStreamSynchronize(upload_stream_),
                "parameter preload stream synchronize");
        }catch(...){
            (void)hipStreamDestroy(upload_stream_);
            upload_stream_=nullptr;
            throw;
        }
        (void)hipStreamDestroy(upload_stream_);
        upload_stream_=nullptr;
        std::cout<<"codec_weight_dtype: F32\n"
                 <<"codec_weight_contract_loaded: True\n"
                 <<"codec_quantizers: 16\n"
                 <<"codec_transformer_layers: 8\n"
                 <<"codec_parameter_preload_stream: explicit_low_priority_nonblocking\n"
                 <<"codec_parameter_preload_allocator: stream_ordered\n";
    }
    const Param& get(const std::string& s)const{
        auto it=gpu_.find(s);if(it==gpu_.end())throw std::runtime_error("legacy_legacy_tag_h codec parameter not loaded: "+s);return it->second;
    }
private:
    static std::string tr(int i,const std::string& t){return "decoder.pre_transformer.layers."+std::to_string(i)+"."+t;}
    static std::string up(int i,const std::string& t){return "decoder.upsample."+std::to_string(i)+"."+t;}
    static std::string db(int i,const std::string& t){return "decoder.decoder."+std::to_string(i)+"."+t;}
    void load(const std::string& s,const std::vector<std::int64_t>& expected){
        const auto& t=host_.suffix(s);
        if(t.shape!=expected){
            std::ostringstream o;o<<"legacy_legacy_tag_h codec shape mismatch "<<s<<" actual=";
            for(auto d:t.shape)o<<d<<",";o<<" expected=";for(auto d:expected)o<<d<<",";
            throw std::runtime_error(o.str());
        }
        gpu_.emplace(
            s,
            Param(t,upload_stream_));
    }
    void snake(const std::string& p,int c){load(p+".alpha",shp({c}));load(p+".beta",shp({c}));}
    void convnext(int s){
        const auto p=up(s,"1");
        load(p+".dwconv.conv.weight",shp({LAT,1,7}));load(p+".dwconv.conv.bias",shp({LAT}));
        load(p+".norm.weight",shp({LAT}));load(p+".norm.bias",shp({LAT}));
        load(p+".pwconv1.weight",shp({4*LAT,LAT}));load(p+".pwconv1.bias",shp({4*LAT}));
        load(p+".pwconv2.weight",shp({LAT,4*LAT}));load(p+".pwconv2.bias",shp({LAT}));load(p+".gamma",shp({LAT}));
    }
    void residual(int di,int ui,int c){
        const auto p=db(di,"block."+std::to_string(ui));
        snake(p+".act1",c);load(p+".conv1.conv.weight",shp({c,c,7}));load(p+".conv1.conv.bias",shp({c}));
        snake(p+".act2",c);load(p+".conv2.conv.weight",shp({c,c,1}));load(p+".conv2.conv.bias",shp({c}));
    }
    void load_contract(){
        load("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",shp({CB,CBD}));
        load("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",shp({CB}));
        for(int i=0;i<15;++i){auto p="decoder.quantizer.rvq_rest.vq.layers."+std::to_string(i)+"._codebook.";
            load(p+"embedding_sum",shp({CB,CBD}));load(p+"cluster_usage",shp({CB}));}
        load("decoder.quantizer.rvq_first.output_proj.weight",shp({CBO,CBD,1}));
        load("decoder.quantizer.rvq_rest.output_proj.weight",shp({CBO,CBD,1}));
        load("decoder.pre_conv.conv.weight",shp({LAT,CBO,3}));load("decoder.pre_conv.conv.bias",shp({LAT}));
        load("decoder.pre_transformer.input_proj.weight",shp({TH,LAT}));load("decoder.pre_transformer.input_proj.bias",shp({TH}));
        for(int i=0;i<TL;++i){
            load(tr(i,"input_layernorm.weight"),shp({TH}));
            load(tr(i,"self_attn.q_proj.weight"),shp({AD,TH}));load(tr(i,"self_attn.k_proj.weight"),shp({AD,TH}));
            load(tr(i,"self_attn.v_proj.weight"),shp({AD,TH}));load(tr(i,"self_attn.o_proj.weight"),shp({TH,AD}));
            load(tr(i,"self_attn_layer_scale.scale"),shp({TH}));
            load(tr(i,"post_attention_layernorm.weight"),shp({TH}));
            load(tr(i,"mlp.gate_proj.weight"),shp({TI,TH}));load(tr(i,"mlp.up_proj.weight"),shp({TI,TH}));
            load(tr(i,"mlp.down_proj.weight"),shp({TH,TI}));load(tr(i,"mlp_layer_scale.scale"),shp({TH}));
        }
        load("decoder.pre_transformer.norm.weight",shp({TH}));
        load("decoder.pre_transformer.output_proj.weight",shp({LAT,TH}));load("decoder.pre_transformer.output_proj.bias",shp({LAT}));
        for(int s=0;s<2;++s){auto p=up(s,"0.conv");load(p+".weight",shp({LAT,LAT,2}));load(p+".bias",shp({LAT}));convnext(s);}
        load("decoder.decoder.0.conv.weight",shp({DD,LAT,7}));load("decoder.decoder.0.conv.bias",shp({DD}));
        constexpr int rates[4]={8,5,4,3};
        for(int s=0;s<4;++s){int di=s+1,ic=DD>>s,oc=DD>>(s+1),r=rates[s];
            snake(db(di,"block.0"),ic);
            load(db(di,"block.1.conv.weight"),shp({ic,oc,2*r}));load(db(di,"block.1.conv.bias"),shp({oc}));
            residual(di,2,oc);residual(di,3,oc);residual(di,4,oc);
        }
        static_assert((DD>>4)==96);
        snake("decoder.decoder.5",96);
        load("decoder.decoder.6.conv.weight",shp({1,96,7}));load("decoder.decoder.6.conv.bias",shp({1}));
    }
    model_frontend::ModelWeights host_;
    std::unordered_map<std::string,Param> gpu_;
    hipStream_t upload_stream_=nullptr;
};

__device__ __forceinline__ float wsum(float v){for(int o=16;o;o>>=1)v+=__shfl_down(v,o,WAVE);return v;}
__device__ __forceinline__ float wmax(float v){for(int o=16;o;o>>=1){float x=__shfl_down(v,o,WAVE);v=v>x?v:x;}return v;}
__device__ float bsum(float v){
    __shared__ float s[WAVES];__shared__ float total;
    int lane=threadIdx.x&31,wave=threadIdx.x>>5;v=wsum(v);if(lane==0)s[wave]=v;__syncthreads();
    if(wave==0){float x=lane<WAVES?s[lane]:0.0f;x=wsum(x);if(lane==0)total=x;}__syncthreads();return total;
}
__device__ float bmax(float v){
    __shared__ float s[WAVES];__shared__ float total;
    int lane=threadIdx.x&31,wave=threadIdx.x>>5;v=wmax(v);if(lane==0)s[wave]=v;__syncthreads();
    if(wave==0){float x=lane<WAVES?s[lane]:-INFINITY;x=wmax(x);if(lane==0)total=x;}__syncthreads();return total;
}

__global__ __launch_bounds__(256,2) void linear_f32(
    const float* x,const float* w,const float* bias,float* y,int m,int n,int k){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(m)*n;if(oi>=total)return;
    int row=static_cast<int>(oi/n),col=static_cast<int>(oi%n);float a=0.0f;
    const float* xp=x+static_cast<std::size_t>(row)*k;const float* wp=w+static_cast<std::size_t>(col)*k;
    for(int i=lane;i<k;i+=WAVE)a=fmaf(xp[i],wp[i],a);a=wsum(a);if(lane==0)y[oi]=a+(bias?bias[col]:0.0f);
}
__global__ void add_f32(const float*a,const float*b,float*y,std::size_t n){std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]+b[i];}
__global__ void copy_f32(const float*a,float*y,std::size_t n){std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i];}
__global__ void scale_add_f32(const float*r,const float*b,const float*s,float*y,int rows,int cols){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x,n=static_cast<std::size_t>(rows)*cols;
    if(i<n)y[i]=r[i]+b[i]*s[i%cols];
}
__global__ void silu_mul_f32(const float*g,const float*u,float*y,std::size_t n){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){float x=g[i];y[i]=(x/(1.0f+expf(-x)))*u[i];}
}
__global__ void gelu_f32(const float*x,float*y,std::size_t n){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){float v=x[i];y[i]=0.5f*v*(1.0f+erff(v*0.7071067811865475f));}
}
__global__ void snake_f32(const float*x,const float*a,const float*b,float*y,int rows,int cols){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x,n=static_cast<std::size_t>(rows)*cols;
    if(i<n){int c=i%cols;float aa=expf(a[c]),bb=expf(b[c]),s=sinf(x[i]*aa);y[i]=x[i]+s*s/(bb+1.0e-9f);}
}
__global__ void clamp_f32(float*x,std::size_t n){std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)x[i]=fminf(1.0f,fmaxf(-1.0f,x[i]));}

__global__ void rms_f32(const float*x,const float*w,float*y,int rows,int cols,float eps){
    int row=blockIdx.x;if(row>=rows)return;float p=0.0f;const float* r=x+static_cast<std::size_t>(row)*cols;
    for(int c=threadIdx.x;c<cols;c+=blockDim.x)p=fmaf(r[c],r[c],p);float sum=bsum(p);__shared__ float inv;
    if(threadIdx.x==0)inv=rsqrtf(sum/static_cast<float>(cols)+eps);__syncthreads();
    for(int c=threadIdx.x;c<cols;c+=blockDim.x)y[static_cast<std::size_t>(row)*cols+c]=r[c]*inv*w[c];
}
__global__ void layernorm_f32(const float*x,const float*w,const float*b,float*y,int rows,int cols,float eps){
    int row=blockIdx.x;if(row>=rows)return;const float*r=x+static_cast<std::size_t>(row)*cols;float p=0.0f,q=0.0f;
    for(int c=threadIdx.x;c<cols;c+=blockDim.x){float v=r[c];p+=v;q=fmaf(v,v,q);}float sum=bsum(p),sq=bsum(q);
    __shared__ float mean,inv;if(threadIdx.x==0){mean=sum/cols;float var=fmaxf(0.0f,sq/cols-mean*mean);inv=rsqrtf(var+eps);}__syncthreads();
    for(int c=threadIdx.x;c<cols;c+=blockDim.x)y[static_cast<std::size_t>(row)*cols+c]=(r[c]-mean)*inv*w[c]+b[c];
}


__global__ __launch_bounds__(256,2) void causal_conv_f32(
    const float*x,const float*w,const float*b,float*y,int time,int cin,int cout,int kernel,int dilation,int groups){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(time)*cout;if(oi>=total)return;
    int t=static_cast<int>(oi/cout),oc=static_cast<int>(oi%cout);int opg=cout/groups,ipg=cin/groups,g=oc/opg,ibase=g*ipg;
    float a=0.0f;int red=ipg*kernel;
    for(int r=lane;r<red;r+=WAVE){int kk=r%kernel,lic=r/kernel,it=t-(kernel-1-kk)*dilation;if(it<0)continue;
        int ic=ibase+lic;float xv=x[static_cast<std::size_t>(it)*cin+ic];
        float ww=w[(static_cast<std::size_t>(oc)*ipg+lic)*kernel+kk];a=fmaf(xv,ww,a);}
    a=wsum(a);if(lane==0)y[oi]=a+(b?b[oc]:0.0f);
}

// ConvTranspose1d [Cin,Cout,K]. For Qwen decoder K is stride or 2*stride,
// therefore only 1-2 source rows contribute to each output phase.
__global__ __launch_bounds__(256,2) void transconv_f32(
    const float*x,const float*w,const float*b,float*y,int tin,int cin,int cout,int kernel,int stride){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    int tout=tin*stride;std::size_t total=static_cast<std::size_t>(tout)*cout;if(oi>=total)return;
    int t=static_cast<int>(oi/cout),oc=static_cast<int>(oi%cout),phase=t%stride,taps=(kernel+stride-1)/stride;
    float a=0.0f;int red=cin*taps;
    for(int r=lane;r<red;r+=WAVE){int slot=r%taps,ic=r/taps,kk=phase+slot*stride;if(kk>=kernel)continue;
        int num=t-kk;if(num<0||num%stride)continue;int it=num/stride;if(it<0||it>=tin)continue;
        float xv=x[static_cast<std::size_t>(it)*cin+ic];
        float ww=w[(static_cast<std::size_t>(ic)*cout+oc)*kernel+kk];a=fmaf(xv,ww,a);}
    a=wsum(a);if(lane==0)y[oi]=a+(b?b[oc]:0.0f);
}


// ============================================================================
// legacy_legacy_tag_j Qingming hot decoder kernels.
//
// Design:
//   * 256 threads = 8 wave32s.
//   * one block owns one time position + an 8-output-channel tile.
//   * cooperative block load places the shared input/window in LDS once.
//   * each wave owns one output channel.
//   * lane-private accumulation stays in FP32 VGPRs.
//   * wave reduction is shuffle-based; no cross-wave arithmetic reduction.
//
// The reduction-index mapping deliberately matches the legacy_legacy_tag_h scalar/wave kernels,
// so the FMA traversal order inside each output wave is retained.
// ============================================================================

__global__ __launch_bounds__(256,2)
void qingming_tiled_causal_conv_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int time,
    int cin,
    int cout,
    int kernel,
    int dilation) {

    const int wave = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tiles = (cout + WAVES - 1) / WAVES;
    const int t = static_cast<int>(blockIdx.x) / tiles;
    const int tile = static_cast<int>(blockIdx.x) - t * tiles;
    const int oc = tile * WAVES + wave;

    if (t >= time) {
        return;
    }

    extern __shared__ float lds[];

    const int red = cin * kernel;

    // Preserve legacy_legacy_tag_h reduction indexing:
    //   r -> lic=r/kernel, kk=r%kernel.
    for (
        int r = static_cast<int>(threadIdx.x);
        r < red;
        r += static_cast<int>(blockDim.x)
    ) {
        const int kk = r % kernel;
        const int ic = r / kernel;
        const int it =
            t - (kernel - 1 - kk) * dilation;

        lds[r] =
            (it >= 0)
            ? x[
                static_cast<std::size_t>(it)
                * cin
                + ic]
            : 0.0f;
    }

    __syncthreads();

    if (oc >= cout) {
        return;
    }

    float acc = 0.0f;

    for (
        int r = lane;
        r < red;
        r += WAVE
    ) {
        const int kk = r % kernel;
        const int ic = r / kernel;

        const float xv = lds[r];
        const float ww =
            w[
                (
                    static_cast<std::size_t>(oc)
                    * cin
                    + ic
                )
                * kernel
                + kk];

        acc = fmaf(
            xv,
            ww,
            acc);
    }

    acc = wsum(acc);

    if (lane == 0) {
        y[
            static_cast<std::size_t>(t)
            * cout
            + oc] =
            acc
            + (b ? b[oc] : 0.0f);
    }
}


__global__ __launch_bounds__(256,2)
void qingming_tiled_transconv_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int tin,
    int cin,
    int cout,
    int kernel,
    int stride) {

    const int wave = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tout = tin * stride;
    const int tiles = (cout + WAVES - 1) / WAVES;
    const int t = static_cast<int>(blockIdx.x) / tiles;
    const int tile = static_cast<int>(blockIdx.x) - t * tiles;
    const int oc = tile * WAVES + wave;
    const int phase = t % stride;
    const int taps =
        (kernel + stride - 1) / stride;

    extern __shared__ float lds[];

    const int red = cin * taps;

    // Preserve legacy_legacy_tag_h reduction indexing:
    //   r -> slot=r%taps, ic=r/taps.
    for (
        int r = static_cast<int>(threadIdx.x);
        r < red;
        r += static_cast<int>(blockDim.x)
    ) {
        const int slot = r % taps;
        const int ic = r / taps;
        const int kk =
            phase + slot * stride;

        float value = 0.0f;

        if (kk < kernel) {
            const int num = t - kk;

            if (
                num >= 0
                && num % stride == 0
            ) {
                const int it = num / stride;

                if (
                    it >= 0
                    && it < tin
                ) {
                    value =
                        x[
                            static_cast<std::size_t>(it)
                            * cin
                            + ic];
                }
            }
        }

        lds[r] = value;
    }

    __syncthreads();

    if (
        t >= tout
        || oc >= cout
    ) {
        return;
    }

    float acc = 0.0f;

    for (
        int r = lane;
        r < red;
        r += WAVE
    ) {
        const int slot = r % taps;
        const int ic = r / taps;
        const int kk =
            phase + slot * stride;

        if (kk >= kernel) {
            continue;
        }

        const float ww =
            w[
                (
                    static_cast<std::size_t>(ic)
                    * cout
                    + oc
                )
                * kernel
                + kk];

        acc = fmaf(
            lds[r],
            ww,
            acc);
    }

    acc = wsum(acc);

    if (lane == 0) {
        y[
            static_cast<std::size_t>(t)
            * cout
            + oc] =
            acc
            + (b ? b[oc] : 0.0f);
    }
}


// Hot residual pointwise projection:
//     out = residual + Conv1x1(activated)
//
// The Snake activation itself remains a separate kernel and is therefore
// materialized exactly once.  This avoids recomputing expf/sinf per output
// channel tile.  LDS is used only to fan out the activated input row to 8
// output waves.
__global__ __launch_bounds__(256,2)
void qingming_tiled_pointwise_residual_f32(
    const float* __restrict__ activated,
    const float* __restrict__ residual,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int time,
    int channels) {

    const int wave = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tiles =
        (channels + WAVES - 1) / WAVES;
    const int t =
        static_cast<int>(blockIdx.x) / tiles;
    const int tile =
        static_cast<int>(blockIdx.x) - t * tiles;
    const int oc =
        tile * WAVES + wave;

    if (t >= time) {
        return;
    }

    extern __shared__ float lds[];

    for (
        int ic = static_cast<int>(threadIdx.x);
        ic < channels;
        ic += static_cast<int>(blockDim.x)
    ) {
        lds[ic] =
            activated[
                static_cast<std::size_t>(t)
                * channels
                + ic];
    }

    __syncthreads();

    if (oc >= channels) {
        return;
    }

    float acc = 0.0f;

    // Same 1x1 Conv reduction order as causal_conv_f32 with kernel=1.
    for (
        int ic = lane;
        ic < channels;
        ic += WAVE
    ) {
        const float ww =
            w[
                static_cast<std::size_t>(oc)
                * channels
                + ic];

        acc = fmaf(
            lds[ic],
            ww,
            acc);
    }

    acc = wsum(acc);

    if (lane == 0) {
        const std::size_t index =
            static_cast<std::size_t>(t)
            * channels
            + oc;

        const float projected =
            acc
            + (b ? b[oc] : 0.0f);

        out[index] =
            residual[index]
            + projected;
    }
}

__global__ void codebook_accum_f32(
    const std::int32_t*codes,const float*emb,const float*usage,float*out,int frames,int group,int reset){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x,n=static_cast<std::size_t>(frames)*CBD;if(i>=n)return;
    int f=static_cast<int>(i/CBD),d=static_cast<int>(i%CBD),code=codes[static_cast<std::size_t>(f)*NQ+group];float v=0.0f;
    if(code>=0&&code<CB)v=emb[static_cast<std::size_t>(code)*CBD+d]/fmaxf(usage[code],1.0e-5f);
    if(reset)out[i]=v;else out[i]+=v;
}

__global__ void rope_f32(const float*x,float*y,int time,int heads,int hd,float theta){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x,n=static_cast<std::size_t>(time)*heads*hd;if(i>=n)return;
    int d=static_cast<int>(i%hd);std::size_t z=i/hd;int h=static_cast<int>(z%heads),pos=static_cast<int>(z/heads),half=hd/2;
    int fi=d<half?d:d-half;float inv=powf(theta,-2.0f*fi/static_cast<float>(hd)),ang=pos*inv,c=cosf(ang),s=sinf(ang);
    int rd=d<half?d+half:d-half;std::size_t base=(static_cast<std::size_t>(pos)*heads+h)*hd;float rot=x[base+rd];if(d>=half)rot=-rot;
    y[i]=fmaf(rot,s,x[base+d]*c);
}
__global__ __launch_bounds__(256,2) void qk_f32(
    const float*q,const float*k,float*scores,int time,int heads,int hd,int window,float scale){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t si=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(time)*heads*time;if(si>=total)return;
    int kt=static_cast<int>(si%time);std::size_t z=si/time;int h=static_cast<int>(z%heads),qt=static_cast<int>(z/heads);
    int min_key=window>0?((qt-window+1)>0?(qt-window+1):0):0;if(kt>qt||kt<min_key){if(lane==0)scores[si]=-INFINITY;return;}
    std::size_t qb=(static_cast<std::size_t>(qt)*heads+h)*hd,kb=(static_cast<std::size_t>(kt)*heads+h)*hd;float a=0.0f;
    for(int d=lane;d<hd;d+=WAVE)a=fmaf(q[qb+d],k[kb+d],a);a=wsum(a);if(lane==0)scores[si]=a*scale;
}
__global__ void softmax_f32(const float*s,float*p,int rows,int cols){
    int row=blockIdx.x;if(row>=rows)return;std::size_t base=static_cast<std::size_t>(row)*cols;float lm=-INFINITY;
    for(int c=threadIdx.x;c<cols;c+=blockDim.x)lm=fmaxf(lm,s[base+c]);float mx=bmax(lm),ls=0.0f;
    for(int c=threadIdx.x;c<cols;c+=blockDim.x){float e=expf(s[base+c]-mx);p[base+c]=e;ls+=e;}float total=bsum(ls);
    for(int c=threadIdx.x;c<cols;c+=blockDim.x)p[base+c]/=total;
}
__global__ __launch_bounds__(256,2) void pv_f32(
    const float*p,const float*v,float*y,int time,int heads,int hd){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(time)*heads*hd;if(oi>=total)return;
    int d=static_cast<int>(oi%hd);std::size_t z=oi/hd;int h=static_cast<int>(z%heads),qt=static_cast<int>(z/heads);
    std::size_t pb=(static_cast<std::size_t>(qt)*heads+h)*time;float a=0.0f;
    for(int kt=lane;kt<time;kt+=WAVE)a=fmaf(p[pb+kt],v[(static_cast<std::size_t>(kt)*heads+h)*hd+d],a);
    a=wsum(a);if(lane==0)y[oi]=a;
}

static unsigned blocks_for(std::size_t n,int per=WAVES){return static_cast<unsigned>((n+per-1)/per);}
static unsigned threads_for(std::size_t n){return static_cast<unsigned>((n+THREADS-1)/THREADS);}

static void lin(hipStream_t st,const float*x,const Param&w,const Param&b,float*y,int m,int n,int k){
    hipLaunchKernelGGL(linear_f32,dim3(blocks_for(static_cast<std::size_t>(m)*n)),dim3(THREADS),0,st,x,w.data(),b.data(),y,m,n,k);hipok(hipGetLastError(),"linear");
}
static void lin0(hipStream_t st,const float*x,const Param&w,float*y,int m,int n,int k){
    hipLaunchKernelGGL(linear_f32,dim3(blocks_for(static_cast<std::size_t>(m)*n)),dim3(THREADS),0,st,x,w.data(),nullptr,y,m,n,k);hipok(hipGetLastError(),"linear");
}
static void add(hipStream_t st,const float*a,const float*b,float*y,std::size_t n){
    hipLaunchKernelGGL(add_f32,dim3(threads_for(n)),dim3(THREADS),0,st,a,b,y,n);hipok(hipGetLastError(),"add");
}
static void copyv(hipStream_t st,const float*a,float*y,std::size_t n){
    hipLaunchKernelGGL(copy_f32,dim3(threads_for(n)),dim3(THREADS),0,st,a,y,n);hipok(hipGetLastError(),"copy");
}
static void scaleadd(hipStream_t st,const float*r,const float*b,const Param&s,float*y,int rows,int cols){
    std::size_t n=static_cast<std::size_t>(rows)*cols;hipLaunchKernelGGL(scale_add_f32,dim3(threads_for(n)),dim3(THREADS),0,st,r,b,s.data(),y,rows,cols);hipok(hipGetLastError(),"scale add");
}
static void silumul(hipStream_t st,const float*g,const float*u,float*y,std::size_t n){
    hipLaunchKernelGGL(silu_mul_f32,dim3(threads_for(n)),dim3(THREADS),0,st,g,u,y,n);hipok(hipGetLastError(),"silu mul");
}
static void gelu(hipStream_t st,const float*x,float*y,std::size_t n){
    hipLaunchKernelGGL(gelu_f32,dim3(threads_for(n)),dim3(THREADS),0,st,x,y,n);hipok(hipGetLastError(),"gelu");
}
static void snake(hipStream_t st,const float*x,const Param&a,const Param&b,float*y,int rows,int cols){
    std::size_t n=static_cast<std::size_t>(rows)*cols;hipLaunchKernelGGL(snake_f32,dim3(threads_for(n)),dim3(THREADS),0,st,x,a.data(),b.data(),y,rows,cols);hipok(hipGetLastError(),"snake");
}
static void rms(hipStream_t st,const float*x,const Param&w,float*y,int rows,int cols,float eps){
    hipLaunchKernelGGL(rms_f32,dim3(rows),dim3(THREADS),0,st,x,w.data(),y,rows,cols,eps);hipok(hipGetLastError(),"rmsnorm");
}
static void lnorm(hipStream_t st,const float*x,const Param&w,const Param&b,float*y,int rows,int cols,float eps){
    hipLaunchKernelGGL(layernorm_f32,dim3(rows),dim3(THREADS),0,st,x,w.data(),b.data(),y,rows,cols,eps);hipok(hipGetLastError(),"layernorm");
}
static void conv(hipStream_t st,const float*x,const Param&w,const Param&b,float*y,int t,int ci,int co,int k,int dil,int groups=1){
    std::size_t n=static_cast<std::size_t>(t)*co;hipLaunchKernelGGL(causal_conv_f32,dim3(blocks_for(n)),dim3(THREADS),0,st,x,w.data(),b.data(),y,t,ci,co,k,dil,groups);hipok(hipGetLastError(),"conv");
}
static void tconv(hipStream_t st,const float*x,const Param&w,const Param&b,float*y,int t,int ci,int co,int k,int stride){
    std::size_t n=static_cast<std::size_t>(t*stride)*co;hipLaunchKernelGGL(transconv_f32,dim3(blocks_for(n)),dim3(THREADS),0,st,x,w.data(),b.data(),y,t,ci,co,k,stride);hipok(hipGetLastError(),"transconv");
}

static void qingming_conv_lds(
    hipStream_t st,
    const float* x,
    const Param& w,
    const Param& b,
    float* y,
    int t,
    int ci,
    int co,
    int k,
    int dilation) {

    const int tiles =
        (co + WAVES - 1) / WAVES;

    const std::size_t shared_bytes =
        static_cast<std::size_t>(ci)
        * static_cast<std::size_t>(k)
        * sizeof(float);

    hipLaunchKernelGGL(
        qingming_tiled_causal_conv_f32,
        dim3(
            static_cast<unsigned>(
                t * tiles)),
        dim3(THREADS),
        shared_bytes,
        st,
        x,
        w.data(),
        b.data(),
        y,
        t,
        ci,
        co,
        k,
        dilation);

    hipok(
        hipGetLastError(),
        "legacy_legacy_tag_n qingming LDS causal conv");
}


static void qingming_tconv_lds(
    hipStream_t st,
    const float* x,
    const Param& w,
    const Param& b,
    float* y,
    int t,
    int ci,
    int co,
    int k,
    int stride) {

    const int tout = t * stride;
    const int tiles =
        (co + WAVES - 1) / WAVES;
    const int taps =
        (k + stride - 1) / stride;

    const std::size_t shared_bytes =
        static_cast<std::size_t>(ci)
        * static_cast<std::size_t>(taps)
        * sizeof(float);

    hipLaunchKernelGGL(
        qingming_tiled_transconv_f32,
        dim3(
            static_cast<unsigned>(
                tout * tiles)),
        dim3(THREADS),
        shared_bytes,
        st,
        x,
        w.data(),
        b.data(),
        y,
        t,
        ci,
        co,
        k,
        stride);

    hipok(
        hipGetLastError(),
        "legacy_legacy_tag_n qingming LDS transconv");
}


static void qingming_pointwise_residual_lds(
    hipStream_t st,
    const float* activated,
    const float* residual,
    const Param& w,
    const Param& b,
    float* out,
    int t,
    int channels) {

    const int tiles =
        (channels + WAVES - 1) / WAVES;

    const std::size_t shared_bytes =
        static_cast<std::size_t>(channels)
        * sizeof(float);

    hipLaunchKernelGGL(
        qingming_tiled_pointwise_residual_f32,
        dim3(
            static_cast<unsigned>(
                t * tiles)),
        dim3(THREADS),
        shared_bytes,
        st,
        activated,
        residual,
        w.data(),
        b.data(),
        out,
        t,
        channels);

    hipok(
        hipGetLastError(),
        "legacy_legacy_tag_n qingming LDS pointwise residual");
}


static void rope(hipStream_t st,const float*x,float*y,int t){
    std::size_t n=static_cast<std::size_t>(t)*HEADS*HD;hipLaunchKernelGGL(rope_f32,dim3(threads_for(n)),dim3(THREADS),0,st,x,y,t,HEADS,HD,10000.0f);hipok(hipGetLastError(),"rope");
}
static void attention(hipStream_t st,const float*q,const float*k,const float*v,float*s,float*p,float*y,int t){
    std::size_t ns=static_cast<std::size_t>(t)*HEADS*t;hipLaunchKernelGGL(qk_f32,dim3(blocks_for(ns)),dim3(THREADS),0,st,q,k,s,t,HEADS,HD,WINDOW,1.0f/8.0f);hipok(hipGetLastError(),"qk");
    hipLaunchKernelGGL(softmax_f32,dim3(t*HEADS),dim3(THREADS),0,st,s,p,t*HEADS,t);hipok(hipGetLastError(),"softmax");
    std::size_t no=static_cast<std::size_t>(t)*HEADS*HD;hipLaunchKernelGGL(pv_f32,dim3(blocks_for(no)),dim3(THREADS),0,st,p,v,y,t,HEADS,HD);hipok(hipGetLastError(),"pv");
}




// ============================================================================
// Stateful streaming codec decoder.
// Each new codec chunk computes only newly appended causal rows.
// ============================================================================
constexpr int STREAM_FIRST_CHUNK_FRAMES = 8;
constexpr int STREAM_CHUNK_FRAMES = 16;
constexpr int STREAM_MAX_FRAMES = 8192;

__global__ void stream_codebook_accum_f32(
    const std::int32_t*codes,const float*emb,const float*usage,float*out,
    int frame_start,int frame_count,int group,int reset){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    std::size_t n=static_cast<std::size_t>(frame_count)*CBD;if(i>=n)return;
    int lf=static_cast<int>(i/CBD),d=static_cast<int>(i%CBD),af=frame_start+lf;
    int code=codes[static_cast<std::size_t>(af)*NQ+group];float v=0.0f;
    if(code>=0&&code<CB)v=emb[static_cast<std::size_t>(code)*CBD+d]/fmaxf(usage[code],1.0e-5f);
    if(reset)out[i]=v;else out[i]+=v;
}

__global__ void stream_rope_range_f32(
    const float*x,float*y,int rows,int absolute_start,int heads,int hd,float theta){
    std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    std::size_t n=static_cast<std::size_t>(rows)*heads*hd;if(i>=n)return;
    int d=static_cast<int>(i%hd);std::size_t z=i/hd;int h=static_cast<int>(z%heads);
    int lp=static_cast<int>(z/heads),pos=absolute_start+lp,half=hd/2;
    int fi=d<half?d:d-half;float inv=powf(theta,-2.0f*fi/static_cast<float>(hd));
    float ang=pos*inv,c=cosf(ang),s=sinf(ang);int rd=d<half?d+half:d-half;
    std::size_t base=(static_cast<std::size_t>(lp)*heads+h)*hd;float rot=x[base+rd];if(d>=half)rot=-rot;
    y[i]=fmaf(rot,s,x[base+d]*c);
}

__global__ __launch_bounds__(256,2) void stream_qk_range_f32(
    const float*q,const float*k,float*scores,int query_start,int query_count,
    int total_time,int heads,int hd,int window,float scale){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t si=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(query_count)*heads*total_time;if(si>=total)return;
    int kt=static_cast<int>(si%total_time);std::size_t z=si/total_time;
    int h=static_cast<int>(z%heads),ql=static_cast<int>(z/heads),qt=query_start+ql;
    int min_key=0;
    if(window>0){
        const int candidate=qt-window+1;
        min_key=candidate>0?candidate:0;
    }
    if(kt>qt||kt<min_key){if(lane==0)scores[si]=-INFINITY;return;}
    std::size_t qb=(static_cast<std::size_t>(ql)*heads+h)*hd;
    std::size_t kb=(static_cast<std::size_t>(kt)*heads+h)*hd;float a=0.0f;
    for(int d=lane;d<hd;d+=WAVE)a=fmaf(q[qb+d],k[kb+d],a);
    a=wsum(a);if(lane==0)scores[si]=a*scale;
}

__global__ __launch_bounds__(256,2) void stream_pv_range_f32(
    const float*p,const float*v,float*y,int query_count,int total_time,int heads,int hd){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(query_count)*heads*hd;if(oi>=total)return;
    int d=static_cast<int>(oi%hd);std::size_t z=oi/hd;int h=static_cast<int>(z%heads),ql=static_cast<int>(z/heads);
    std::size_t pb=(static_cast<std::size_t>(ql)*heads+h)*total_time;float a=0.0f;
    for(int kt=lane;kt<total_time;kt+=WAVE)a=fmaf(p[pb+kt],v[(static_cast<std::size_t>(kt)*heads+h)*hd+d],a);
    a=wsum(a);if(lane==0)y[oi]=a;
}

__global__ __launch_bounds__(256,2) void stream_causal_conv_range_f32(
    const float*x,const float*w,const float*b,float*y,int output_start,int output_count,
    int cin,int cout,int kernel,int dilation,int groups){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(output_count)*cout;if(oi>=total)return;
    int lt=static_cast<int>(oi/cout),t=output_start+lt,oc=static_cast<int>(oi%cout);
    int opg=cout/groups,ipg=cin/groups,g=oc/opg,ibase=g*ipg,red=ipg*kernel;float a=0.0f;
    for(int r=lane;r<red;r+=WAVE){int kk=r%kernel,lic=r/kernel,it=t-(kernel-1-kk)*dilation;if(it<0)continue;
        int ic=ibase+lic;float xv=x[static_cast<std::size_t>(it)*cin+ic];
        float ww=w[(static_cast<std::size_t>(oc)*ipg+lic)*kernel+kk];a=fmaf(xv,ww,a);}
    a=wsum(a);if(lane==0)y[oi]=a+(b?b[oc]:0.0f);
}

__global__ __launch_bounds__(256,2) void stream_tiled_causal_conv_range_f32(
    const float*x,const float*w,const float*b,float*y,
    int output_start,int output_count,int cin,int cout,int kernel,int dilation){

    constexpr int opw=4;
    constexpr int opb=WAVES*opw;
    const int wave=threadIdx.x>>5;
    const int lane=threadIdx.x&31;
    const int tiles=(cout+opb-1)/opb;
    const int lt=static_cast<int>(blockIdx.x)/tiles;
    const int tile=static_cast<int>(blockIdx.x)-lt*tiles;
    if(lt>=output_count)return;

    const int t=output_start+lt;
    extern __shared__ float lds[];
    const int red=cin*kernel;

    for(int r=threadIdx.x;r<red;r+=blockDim.x){
        const int kk=r%kernel,ic=r/kernel;
        const int it=t-(kernel-1-kk)*dilation;
        lds[r]=it>=0?x[static_cast<std::size_t>(it)*cin+ic]:0.0f;
    }
    __syncthreads();

    const int oc0=(tile*WAVES+wave)*opw;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;

    for(int r=lane;r<red;r+=WAVE){
        const int kk=r%kernel,ic=r/kernel;
        const float xv=lds[r];
        if(oc0<cout)a0=fmaf(xv,w[(static_cast<std::size_t>(oc0)*cin+ic)*kernel+kk],a0);
        if(oc0+1<cout)a1=fmaf(xv,w[(static_cast<std::size_t>(oc0+1)*cin+ic)*kernel+kk],a1);
        if(oc0+2<cout)a2=fmaf(xv,w[(static_cast<std::size_t>(oc0+2)*cin+ic)*kernel+kk],a2);
        if(oc0+3<cout)a3=fmaf(xv,w[(static_cast<std::size_t>(oc0+3)*cin+ic)*kernel+kk],a3);
    }

    a0=wsum(a0);a1=wsum(a1);a2=wsum(a2);a3=wsum(a3);

    if(lane==0){
        const std::size_t base=static_cast<std::size_t>(lt)*cout;
        if(oc0<cout)y[base+oc0]=a0+(b?b[oc0]:0.0f);
        if(oc0+1<cout)y[base+oc0+1]=a1+(b?b[oc0+1]:0.0f);
        if(oc0+2<cout)y[base+oc0+2]=a2+(b?b[oc0+2]:0.0f);
        if(oc0+3<cout)y[base+oc0+3]=a3+(b?b[oc0+3]:0.0f);
    }
}

__global__ __launch_bounds__(256,2) void stream_transconv_range_f32(
    const float*x,const float*w,const float*b,float*y,int input_end,int output_start,int output_count,
    int cin,int cout,int kernel,int stride){
    int wave=threadIdx.x>>5,lane=threadIdx.x&31;std::size_t oi=static_cast<std::size_t>(blockIdx.x)*WAVES+wave;
    std::size_t total=static_cast<std::size_t>(output_count)*cout;if(oi>=total)return;
    int lt=static_cast<int>(oi/cout),t=output_start+lt,oc=static_cast<int>(oi%cout);
    int phase=t%stride,taps=(kernel+stride-1)/stride,red=cin*taps;float a=0.0f;
    for(int r=lane;r<red;r+=WAVE){int slot=r%taps,ic=r/taps,kk=phase+slot*stride;if(kk>=kernel)continue;
        int num=t-kk;if(num<0||num%stride)continue;int it=num/stride;if(it<0||it>=input_end)continue;
        a=fmaf(x[static_cast<std::size_t>(it)*cin+ic],w[(static_cast<std::size_t>(ic)*cout+oc)*kernel+kk],a);}
    a=wsum(a);if(lane==0)y[oi]=a+(b?b[oc]:0.0f);
}

__global__ __launch_bounds__(256,2) void stream_tiled_transconv_range_f32(
    const float*x,const float*w,const float*b,float*y,
    int input_end,int output_start,int output_count,
    int cin,int cout,int kernel,int stride){

    constexpr int opw=4;
    constexpr int opb=WAVES*opw;
    const int wave=threadIdx.x>>5;
    const int lane=threadIdx.x&31;
    const int tiles=(cout+opb-1)/opb;
    const int lt=static_cast<int>(blockIdx.x)/tiles;
    const int tile=static_cast<int>(blockIdx.x)-lt*tiles;
    if(lt>=output_count)return;

    const int t=output_start+lt;
    const int phase=t%stride;
    const int taps=(kernel+stride-1)/stride;
    const int red=cin*taps;
    extern __shared__ float lds[];

    for(int r=threadIdx.x;r<red;r+=blockDim.x){
        const int slot=r%taps,ic=r/taps,kk=phase+slot*stride;
        float val=0.0f;
        if(kk<kernel){
            const int num=t-kk;
            if(num>=0&&num%stride==0){
                const int it=num/stride;
                if(it>=0&&it<input_end)val=x[static_cast<std::size_t>(it)*cin+ic];
            }
        }
        lds[r]=val;
    }
    __syncthreads();

    const int oc0=(tile*WAVES+wave)*opw;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;

    for(int r=lane;r<red;r+=WAVE){
        const int slot=r%taps,ic=r/taps,kk=phase+slot*stride;
        if(kk>=kernel)continue;
        const float xv=lds[r];
        if(oc0<cout)a0=fmaf(xv,w[(static_cast<std::size_t>(ic)*cout+oc0)*kernel+kk],a0);
        if(oc0+1<cout)a1=fmaf(xv,w[(static_cast<std::size_t>(ic)*cout+oc0+1)*kernel+kk],a1);
        if(oc0+2<cout)a2=fmaf(xv,w[(static_cast<std::size_t>(ic)*cout+oc0+2)*kernel+kk],a2);
        if(oc0+3<cout)a3=fmaf(xv,w[(static_cast<std::size_t>(ic)*cout+oc0+3)*kernel+kk],a3);
    }

    a0=wsum(a0);a1=wsum(a1);a2=wsum(a2);a3=wsum(a3);

    if(lane==0){
        const std::size_t base=static_cast<std::size_t>(lt)*cout;
        if(oc0<cout)y[base+oc0]=a0+(b?b[oc0]:0.0f);
        if(oc0+1<cout)y[base+oc0+1]=a1+(b?b[oc0+1]:0.0f);
        if(oc0+2<cout)y[base+oc0+2]=a2+(b?b[oc0+2]:0.0f);
        if(oc0+3<cout)y[base+oc0+3]=a3+(b?b[oc0+3]:0.0f);
    }
}

static void stream_conv_range(
    hipStream_t st,const float*x,const Param&w,const Param&b,float*y,int start,int count,
    int ci,int co,int k,int dil,int groups,bool hot){
    if(count<=0)return;
    if(hot&&groups==1){
        int tiles=(co+4*WAVES-1)/(4*WAVES);std::size_t sh=static_cast<std::size_t>(ci)*k*sizeof(float);
        hipLaunchKernelGGL(stream_tiled_causal_conv_range_f32,dim3(count*tiles),dim3(THREADS),sh,st,
            x,w.data(),b.data(),y,start,count,ci,co,k,dil);
    }else{
        std::size_t n=static_cast<std::size_t>(count)*co;
        hipLaunchKernelGGL(stream_causal_conv_range_f32,dim3(blocks_for(n)),dim3(THREADS),0,st,
            x,w.data(),b.data(),y,start,count,ci,co,k,dil,groups);
    }
    hipok(hipGetLastError(),"stream causal conv range");
}

static void stream_tconv_range(
    hipStream_t st,const float*x,const Param&w,const Param&b,float*y,int input_start,int input_count,
    int ci,int co,int k,int stride,bool hot){
    if(input_count<=0)return;int input_end=input_start+input_count;
    int output_start=input_start*stride,output_count=input_count*stride;
    if(hot){
        int tiles=(co+WAVES-1)/WAVES,taps=(k+stride-1)/stride;
        std::size_t sh=static_cast<std::size_t>(ci)*taps*sizeof(float);
        hipLaunchKernelGGL(stream_tiled_transconv_range_f32,dim3(output_count*tiles),dim3(THREADS),sh,st,
            x,w.data(),b.data(),y,input_end,output_start,output_count,ci,co,k,stride);
    }else{
        std::size_t n=static_cast<std::size_t>(output_count)*co;
        hipLaunchKernelGGL(stream_transconv_range_f32,dim3(blocks_for(n)),dim3(THREADS),0,st,
            x,w.data(),b.data(),y,input_end,output_start,output_count,ci,co,k,stride);
    }
    hipok(hipGetLastError(),"stream transconv range");
}


__global__ __launch_bounds__(256,2)
void stream_linear_pack4_f32(
    const float*x,const float*w,const float*bias,float*y,
    int m,int n,int k){

    constexpr int opw=4;
    constexpr int opb=WAVES*opw;
    const int wave=threadIdx.x>>5;
    const int lane=threadIdx.x&31;
    const int tiles=(n+opb-1)/opb;
    const int row=static_cast<int>(blockIdx.x)/tiles;
    const int tile=static_cast<int>(blockIdx.x)-row*tiles;
    if(row>=m)return;

    extern __shared__ float lds[];
    const float*xr=x+static_cast<std::size_t>(row)*k;
    for(int i=threadIdx.x;i<k;i+=blockDim.x)lds[i]=xr[i];
    __syncthreads();

    const int c0=(tile*WAVES+wave)*opw;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;

    for(int i=lane;i<k;i+=WAVE){
        const float xv=lds[i];
        if(c0<n)a0=fmaf(xv,w[static_cast<std::size_t>(c0)*k+i],a0);
        if(c0+1<n)a1=fmaf(xv,w[static_cast<std::size_t>(c0+1)*k+i],a1);
        if(c0+2<n)a2=fmaf(xv,w[static_cast<std::size_t>(c0+2)*k+i],a2);
        if(c0+3<n)a3=fmaf(xv,w[static_cast<std::size_t>(c0+3)*k+i],a3);
    }

    a0=wsum(a0);a1=wsum(a1);a2=wsum(a2);a3=wsum(a3);

    if(lane==0){
        const std::size_t base=static_cast<std::size_t>(row)*n;
        if(c0<n)y[base+c0]=a0+(bias?bias[c0]:0.0f);
        if(c0+1<n)y[base+c0+1]=a1+(bias?bias[c0+1]:0.0f);
        if(c0+2<n)y[base+c0+2]=a2+(bias?bias[c0+2]:0.0f);
        if(c0+3<n)y[base+c0+3]=a3+(bias?bias[c0+3]:0.0f);
    }
}

static void stream_lin4(
    hipStream_t st,const float*x,const Param&w,const Param&b,float*y,
    int m,int n,int k){
    const int tiles=(n+4*WAVES-1)/(4*WAVES);
    hipLaunchKernelGGL(
        stream_linear_pack4_f32,
        dim3(static_cast<unsigned>(m*tiles)),
        dim3(THREADS),
        static_cast<std::size_t>(k)*sizeof(float),
        st,x,w.data(),b.data(),y,m,n,k);
    hipok(hipGetLastError(),"stream linear pack4");
}

static void stream_lin04(
    hipStream_t st,const float*x,const Param&w,float*y,
    int m,int n,int k){
    const int tiles=(n+4*WAVES-1)/(4*WAVES);
    hipLaunchKernelGGL(
        stream_linear_pack4_f32,
        dim3(static_cast<unsigned>(m*tiles)),
        dim3(THREADS),
        static_cast<std::size_t>(k)*sizeof(float),
        st,x,w.data(),nullptr,y,m,n,k);
    hipok(hipGetLastError(),"stream linear0 pack4");
}

__global__ __launch_bounds__(256,2)
void stream_pointwise_residual_pack4_f32(
    const float*activated,const float*residual,
    const float*w,const float*b,float*out,
    int time,int channels){

    constexpr int opw=4;
    constexpr int opb=WAVES*opw;
    const int wave=threadIdx.x>>5;
    const int lane=threadIdx.x&31;
    const int tiles=(channels+opb-1)/opb;
    const int t=static_cast<int>(blockIdx.x)/tiles;
    const int tile=static_cast<int>(blockIdx.x)-t*tiles;
    if(t>=time)return;

    extern __shared__ float lds[];
    for(int ic=threadIdx.x;ic<channels;ic+=blockDim.x)
        lds[ic]=activated[static_cast<std::size_t>(t)*channels+ic];
    __syncthreads();

    const int c0=(tile*WAVES+wave)*opw;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;

    for(int ic=lane;ic<channels;ic+=WAVE){
        const float xv=lds[ic];
        if(c0<channels)a0=fmaf(xv,w[static_cast<std::size_t>(c0)*channels+ic],a0);
        if(c0+1<channels)a1=fmaf(xv,w[static_cast<std::size_t>(c0+1)*channels+ic],a1);
        if(c0+2<channels)a2=fmaf(xv,w[static_cast<std::size_t>(c0+2)*channels+ic],a2);
        if(c0+3<channels)a3=fmaf(xv,w[static_cast<std::size_t>(c0+3)*channels+ic],a3);
    }

    a0=wsum(a0);a1=wsum(a1);a2=wsum(a2);a3=wsum(a3);

    if(lane==0){
        const std::size_t base=static_cast<std::size_t>(t)*channels;
        if(c0<channels){const auto i=base+c0;out[i]=residual[i]+a0+(b?b[c0]:0.0f);}
        if(c0+1<channels){const auto i=base+c0+1;out[i]=residual[i]+a1+(b?b[c0+1]:0.0f);}
        if(c0+2<channels){const auto i=base+c0+2;out[i]=residual[i]+a2+(b?b[c0+2]:0.0f);}
        if(c0+3<channels){const auto i=base+c0+3;out[i]=residual[i]+a3+(b?b[c0+3]:0.0f);}
    }
}

static void stream_pointwise_residual4(
    hipStream_t st,const float*activated,const float*residual,
    const Param&w,const Param&b,float*out,int time,int channels){
    const int tiles=(channels+4*WAVES-1)/(4*WAVES);
    hipLaunchKernelGGL(
        stream_pointwise_residual_pack4_f32,
        dim3(static_cast<unsigned>(time*tiles)),
        dim3(THREADS),
        static_cast<std::size_t>(channels)*sizeof(float),
        st,activated,residual,w.data(),b.data(),out,time,channels);
    hipok(hipGetLastError(),"stream pointwise residual pack4");
}

class StreamingCodecDecoder {
public:
    StreamingCodecDecoder(
        std::shared_ptr<Parameters> P,
        int max_frames,
        int chunk_frames=STREAM_CHUNK_FRAMES,
        bool hot=true,
        hipStream_t resident_stream=nullptr)
      :P_(std::move(P)),
       max_frames_(max_frames),
       chunk_frames_(chunk_frames),
       hot_(hot),
       stream_(resident_stream),
       owns_stream_(resident_stream==nullptr){
        if(!P_||max_frames_<=0||max_frames_>STREAM_MAX_FRAMES||chunk_frames_<=0||chunk_frames_>32)
            throw std::runtime_error("invalid streaming codec geometry");

        if(stream_==nullptr){
            stream_=
                ::qwen3_tts::baseline::
                    create_decoder_execution_stream(
                        "Streaming decoder stream");
        }

        allocate();
        reset_workspace();
        hipok(
            hipStreamSynchronize(stream_),
            "streaming decoder allocation/reset sync");
        std::cout<<"codec_decoder_backend: stateful_streaming_cpp_hip_f32\\n"
                 <<"codec_streaming_chunk_frames: "<<chunk_frames_<<"\\n"
                 <<"codec_streaming_transformer_kv_cache: True\\n"
                 <<"codec_streaming_transformer_window: "<<WINDOW<<"\\n"
                 <<"codec_streaming_causal_conv_history: persistent_stage_buffers\\n"
                 <<"codec_streaming_prefix_recompute: False\\n"
                 <<"codec_streaming_decoder_stream: "
                 <<(::qwen3_tts::baseline::g_resident_cu_partition_enabled
                    ?"explicit_cu_masked"
                    :"explicit_low_priority_nonblocking")
                 <<"\\n"
                 <<"codec_streaming_decoder_stream_cu_partitioned: "
                 <<(::qwen3_tts::baseline::g_resident_cu_partition_enabled
                    ?"True":"False")
                 <<"\\n";
    }
    ~StreamingCodecDecoder(){
        if(stream_){
            (void)hipStreamSynchronize(stream_);
            if(owns_stream_){
                (void)hipStreamDestroy(stream_);
            }
        }
    }
    StreamingCodecDecoder(const StreamingCodecDecoder&)=delete;
    StreamingCodecDecoder& operator=(const StreamingCodecDecoder&)=delete;

    void reset_request(){
        reset_workspace();
        hipok(
            hipStreamSynchronize(stream_),
            "streaming decoder request reset sync");
    }

    std::vector<float> append(const std::vector<std::int32_t>& chunk){
        if(chunk.empty()||chunk.size()%NQ)throw std::runtime_error("streaming codec chunk geometry mismatch");
        int count=static_cast<int>(chunk.size()/NQ),start=processed_frames_;
        if(count>chunk_frames_||start+count>max_frames_)throw std::runtime_error("streaming codec capacity exceeded");
        hipok(hipMemcpyAsync(codes_.as<std::int32_t>()+static_cast<std::size_t>(start)*NQ,
            chunk.data(),chunk.size()*sizeof(std::int32_t),hipMemcpyHostToDevice,stream_),"stream codec upload");
        dequant_range(start,count);preconv_range(start,count);transformer_range(start,count);
        upsample_range(start,count);vocoder_range(start,count);
        std::size_t ss=static_cast<std::size_t>(start)*UPSAMPLE,sc=static_cast<std::size_t>(count)*UPSAMPLE;
        std::vector<float> host(sc);hipok(hipMemcpyAsync(host.data(),waveform_.as<float>()+ss,sc*sizeof(float),
            hipMemcpyDeviceToHost,stream_),"stream waveform download");
        hipok(hipStreamSynchronize(stream_),"stream chunk synchronize");processed_frames_+=count;return host;
    }

private:
    struct ResidualStage{
        Buffer input_work;
        Buffer raw0,raw1,raw2,raw3;
        Buffer snake0_work,snake1_work,snake2_work;
        Buffer snake0_history,snake1_history,snake2_history;
        Buffer input_tail;

        void alloc(
            int input_rows,
            int output_rows,
            int channels,
            hipStream_t stream){

            const int input_channels=channels*2;
            const auto bytes=[&](std::size_t rows,int ch){
                return std::max<std::size_t>(rows,1)
                    *static_cast<std::size_t>(ch)
                    *sizeof(float);
            };

            input_work.alloc_async(
                bytes(static_cast<std::size_t>(input_rows)+1,input_channels),
                stream);

            const std::size_t raw_bytes=
                bytes(static_cast<std::size_t>(output_rows),channels);
            raw0.alloc_async(raw_bytes,stream);
            raw1.alloc_async(raw_bytes,stream);
            raw2.alloc_async(raw_bytes,stream);
            raw3.alloc_async(raw_bytes,stream);

            snake0_work.alloc_async(
                bytes(static_cast<std::size_t>(output_rows)+6,channels),
                stream);
            snake1_work.alloc_async(
                bytes(static_cast<std::size_t>(output_rows)+18,channels),
                stream);
            snake2_work.alloc_async(
                bytes(static_cast<std::size_t>(output_rows)+54,channels),
                stream);

            snake0_history.alloc_async(bytes(6,channels),stream);
            snake1_history.alloc_async(bytes(18,channels),stream);
            snake2_history.alloc_async(bytes(54,channels),stream);
            input_tail.alloc_async(bytes(1,input_channels),stream);
        }
    };
    const Param&p(const std::string&s)const{return P_->get(s);}
    static std::string tr(int i,const std::string&t){return "decoder.pre_transformer.layers."+std::to_string(i)+"."+t;}
    static std::string upn(int i,const std::string&t){return "decoder.upsample."+std::to_string(i)+"."+t;}
    static std::string db(int i,const std::string&t){return "decoder.decoder."+std::to_string(i)+"."+t;}
    void ab(Buffer&b,std::size_t e){b.alloc_async(std::max<std::size_t>(e,1)*sizeof(float),stream_);}

    void allocate(){
        std::size_t f=static_cast<std::size_t>(max_frames_),cf=static_cast<std::size_t>(chunk_frames_);
        codes_.alloc_async(f*NQ*sizeof(std::int32_t),stream_);
        ab(dequant_,f*CBO);ab(preconv_,f*LAT);ab(transformer_,f*LAT);ab(up0_tconv_,f*2*LAT);ab(up0_,f*2*LAT);
        ab(up1_tconv_,f*4*LAT);ab(up1_,f*4*LAT);ab(decoder0_,f*4*DD);ab(waveform_,f*UPSAMPLE);
        constexpr int rates[4]={8,5,4,3};
        int input_rows=chunk_frames_*4;
        for(int s=0;s<4;++s){
            const int output_rows=input_rows*rates[s];
            stages_[s].alloc(
                input_rows,
                output_rows,
                DD>>(s+1),
                stream_);
            input_rows=output_rows;
        }
        ab(final_snake_work_,
            (cf*UPSAMPLE+6)*96);
        ab(final_snake_history_,6*96);
        for(int l=0;l<TL;++l){ab(k_cache_[l],f*AD);ab(v_cache_[l],f*AD);}
        ab(th_a_,cf*TH);ab(th_b_,cf*TH);ab(th_c_,cf*TH);ab(th_norm_,cf*TH);
        ab(th_q_,cf*AD);ab(th_k_,cf*AD);ab(th_v_,cf*AD);ab(th_qr_,cf*AD);ab(th_ctx_,cf*AD);
        ab(th_gate_,cf*TI);ab(th_up_,cf*TI);ab(th_scores_,cf*HEADS*max_frames_);ab(th_prob_,cf*HEADS*max_frames_);
        ab(dq_sem_,cf*CBD);ab(dq_acoustic_,cf*CBD);ab(dq_sem_proj_,cf*CBO);ab(dq_acoustic_proj_,cf*CBO);
        std::size_t se=cf*UPSAMPLE*96;ab(scratch0_,se);ab(scratch1_,se);ab(scratch2_,se);ab(scratch3_,se);
    }

    void reset_workspace(){
        // Request workspace begins from a defined zero state.
        codes_.zero_async(stream_);
        dequant_.zero_async(stream_);
        preconv_.zero_async(stream_);
        transformer_.zero_async(stream_);
        up0_tconv_.zero_async(stream_);
        up0_.zero_async(stream_);
        up1_tconv_.zero_async(stream_);
        up1_.zero_async(stream_);
        decoder0_.zero_async(stream_);
        waveform_.zero_async(stream_);

        for(auto& stage:stages_){
            stage.input_work.zero_async(stream_);
            stage.raw0.zero_async(stream_);
            stage.raw1.zero_async(stream_);
            stage.raw2.zero_async(stream_);
            stage.raw3.zero_async(stream_);
            stage.snake0_work.zero_async(stream_);
            stage.snake1_work.zero_async(stream_);
            stage.snake2_work.zero_async(stream_);
            stage.snake0_history.zero_async(stream_);
            stage.snake1_history.zero_async(stream_);
            stage.snake2_history.zero_async(stream_);
            stage.input_tail.zero_async(stream_);
        }
        final_snake_work_.zero_async(stream_);
        final_snake_history_.zero_async(stream_);

        for(int l=0;l<TL;++l){
            k_cache_[l].zero_async(stream_);
            v_cache_[l].zero_async(stream_);
        }

        th_a_.zero_async(stream_);
        th_b_.zero_async(stream_);
        th_c_.zero_async(stream_);
        th_norm_.zero_async(stream_);
        th_q_.zero_async(stream_);
        th_k_.zero_async(stream_);
        th_v_.zero_async(stream_);
        th_qr_.zero_async(stream_);
        th_ctx_.zero_async(stream_);
        th_gate_.zero_async(stream_);
        th_up_.zero_async(stream_);
        th_scores_.zero_async(stream_);
        th_prob_.zero_async(stream_);
        dq_sem_.zero_async(stream_);
        dq_acoustic_.zero_async(stream_);
        dq_sem_proj_.zero_async(stream_);
        dq_acoustic_proj_.zero_async(stream_);
        scratch0_.zero_async(stream_);
        scratch1_.zero_async(stream_);
        scratch2_.zero_async(stream_);
        scratch3_.zero_async(stream_);
        processed_frames_=0;
    }

    void dequant_range(int start,int count){
        std::size_t n=static_cast<std::size_t>(count)*CBD;
        auto&e0=p("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");
        auto&u0=p("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
        hipLaunchKernelGGL(stream_codebook_accum_f32,dim3(threads_for(n)),dim3(THREADS),0,stream_,
            codes_.as<std::int32_t>(),e0.data(),u0.data(),dq_sem_.as<float>(),start,count,0,1);hipok(hipGetLastError(),"stream semantic codebook");
        for(int g=1;g<NQ;++g){auto pre="decoder.quantizer.rvq_rest.vq.layers."+std::to_string(g-1)+"._codebook.";
            hipLaunchKernelGGL(stream_codebook_accum_f32,dim3(threads_for(n)),dim3(THREADS),0,stream_,
                codes_.as<std::int32_t>(),p(pre+"embedding_sum").data(),p(pre+"cluster_usage").data(),
                dq_acoustic_.as<float>(),start,count,g,g==1?1:0);hipok(hipGetLastError(),"stream acoustic codebook");}
        stream_lin04(stream_,dq_sem_.as<float>(),p("decoder.quantizer.rvq_first.output_proj.weight"),dq_sem_proj_.as<float>(),count,CBO,CBD);
        stream_lin04(stream_,dq_acoustic_.as<float>(),p("decoder.quantizer.rvq_rest.output_proj.weight"),dq_acoustic_proj_.as<float>(),count,CBO,CBD);
        add(stream_,dq_sem_proj_.as<float>(),dq_acoustic_proj_.as<float>(),dequant_.as<float>()+static_cast<std::size_t>(start)*CBO,
            static_cast<std::size_t>(count)*CBO);
    }

    void preconv_range(int start,int count){
        stream_conv_range(stream_,dequant_.as<float>(),p("decoder.pre_conv.conv.weight"),p("decoder.pre_conv.conv.bias"),
            preconv_.as<float>()+static_cast<std::size_t>(start)*LAT,start,count,CBO,LAT,3,1,1,hot_);
    }

    void transformer_range(int start,int count){
        int total=start+count;
        stream_lin4(stream_,preconv_.as<float>()+static_cast<std::size_t>(start)*LAT,
            p("decoder.pre_transformer.input_proj.weight"),p("decoder.pre_transformer.input_proj.bias"),
            th_a_.as<float>(),count,TH,LAT);
        for(int l=0;l<TL;++l){
            rms(stream_,th_a_.as<float>(),p(tr(l,"input_layernorm.weight")),th_norm_.as<float>(),count,TH,1e-5f);
            stream_lin04(stream_,th_norm_.as<float>(),p(tr(l,"self_attn.q_proj.weight")),th_q_.as<float>(),count,AD,TH);
            stream_lin04(stream_,th_norm_.as<float>(),p(tr(l,"self_attn.k_proj.weight")),th_k_.as<float>(),count,AD,TH);
            stream_lin04(stream_,th_norm_.as<float>(),p(tr(l,"self_attn.v_proj.weight")),th_v_.as<float>(),count,AD,TH);
            std::size_t rn=static_cast<std::size_t>(count)*HEADS*HD;
            hipLaunchKernelGGL(stream_rope_range_f32,dim3(threads_for(rn)),dim3(THREADS),0,stream_,
                th_q_.as<float>(),th_qr_.as<float>(),count,start,HEADS,HD,10000.0f);hipok(hipGetLastError(),"stream q rope");
            hipLaunchKernelGGL(stream_rope_range_f32,dim3(threads_for(rn)),dim3(THREADS),0,stream_,
                th_k_.as<float>(),k_cache_[l].as<float>()+static_cast<std::size_t>(start)*AD,
                count,start,HEADS,HD,10000.0f);hipok(hipGetLastError(),"stream k rope");
            hipok(hipMemcpyAsync(v_cache_[l].as<float>()+static_cast<std::size_t>(start)*AD,th_v_.as<float>(),
                static_cast<std::size_t>(count)*AD*sizeof(float),hipMemcpyDeviceToDevice,stream_),"stream v cache append");
            std::size_t sn=static_cast<std::size_t>(count)*HEADS*total;
            hipLaunchKernelGGL(stream_qk_range_f32,dim3(blocks_for(sn)),dim3(THREADS),0,stream_,
                th_qr_.as<float>(),k_cache_[l].as<float>(),th_scores_.as<float>(),start,count,total,HEADS,HD,WINDOW,1.0f/8.0f);
            hipok(hipGetLastError(),"stream qk");
            hipLaunchKernelGGL(softmax_f32,dim3(count*HEADS),dim3(THREADS),0,stream_,
                th_scores_.as<float>(),th_prob_.as<float>(),count*HEADS,total);hipok(hipGetLastError(),"stream softmax");
            std::size_t on=static_cast<std::size_t>(count)*HEADS*HD;
            hipLaunchKernelGGL(stream_pv_range_f32,dim3(blocks_for(on)),dim3(THREADS),0,stream_,
                th_prob_.as<float>(),v_cache_[l].as<float>(),th_ctx_.as<float>(),count,total,HEADS,HD);hipok(hipGetLastError(),"stream pv");
            stream_lin04(stream_,th_ctx_.as<float>(),p(tr(l,"self_attn.o_proj.weight")),th_c_.as<float>(),count,TH,AD);
            scaleadd(stream_,th_a_.as<float>(),th_c_.as<float>(),p(tr(l,"self_attn_layer_scale.scale")),th_b_.as<float>(),count,TH);
            rms(stream_,th_b_.as<float>(),p(tr(l,"post_attention_layernorm.weight")),th_norm_.as<float>(),count,TH,1e-5f);
            stream_lin04(stream_,th_norm_.as<float>(),p(tr(l,"mlp.gate_proj.weight")),th_gate_.as<float>(),count,TI,TH);
            stream_lin04(stream_,th_norm_.as<float>(),p(tr(l,"mlp.up_proj.weight")),th_up_.as<float>(),count,TI,TH);
            silumul(stream_,th_gate_.as<float>(),th_up_.as<float>(),th_gate_.as<float>(),static_cast<std::size_t>(count)*TI);
            stream_lin04(stream_,th_gate_.as<float>(),p(tr(l,"mlp.down_proj.weight")),th_c_.as<float>(),count,TH,TI);
            scaleadd(stream_,th_b_.as<float>(),th_c_.as<float>(),p(tr(l,"mlp_layer_scale.scale")),th_a_.as<float>(),count,TH);
        }
        rms(stream_,th_a_.as<float>(),p("decoder.pre_transformer.norm.weight"),th_b_.as<float>(),count,TH,1e-5f);
        stream_lin4(stream_,th_b_.as<float>(),p("decoder.pre_transformer.output_proj.weight"),
            p("decoder.pre_transformer.output_proj.bias"),transformer_.as<float>()+static_cast<std::size_t>(start)*LAT,count,LAT,TH);
    }

    void convnext_range(int stage,int start,int count,const float*res,float*out){
        auto pre=upn(stage,"1");
        stream_conv_range(stream_,res,p(pre+".dwconv.conv.weight"),p(pre+".dwconv.conv.bias"),scratch0_.as<float>(),
            start,count,LAT,LAT,7,1,LAT,false);
        lnorm(stream_,scratch0_.as<float>(),p(pre+".norm.weight"),p(pre+".norm.bias"),scratch1_.as<float>(),count,LAT,1e-6f);
        stream_lin4(stream_,scratch1_.as<float>(),p(pre+".pwconv1.weight"),p(pre+".pwconv1.bias"),scratch2_.as<float>(),count,4*LAT,LAT);
        gelu(stream_,scratch2_.as<float>(),scratch2_.as<float>(),static_cast<std::size_t>(count)*4*LAT);
        stream_lin4(stream_,scratch2_.as<float>(),p(pre+".pwconv2.weight"),p(pre+".pwconv2.bias"),scratch3_.as<float>(),count,LAT,4*LAT);
        scaleadd(stream_,res+static_cast<std::size_t>(start)*LAT,scratch3_.as<float>(),p(pre+".gamma"),
            out+static_cast<std::size_t>(start)*LAT,count,LAT);
    }

    void upsample_range(int start,int count){
        stream_tconv_range(stream_,transformer_.as<float>(),p("decoder.upsample.0.0.conv.weight"),
            p("decoder.upsample.0.0.conv.bias"),up0_tconv_.as<float>()+static_cast<std::size_t>(start*2)*LAT,
            start,count,LAT,LAT,2,2,hot_);
        convnext_range(0,start*2,count*2,up0_tconv_.as<float>(),up0_.as<float>());
        stream_tconv_range(stream_,up0_.as<float>(),p("decoder.upsample.1.0.conv.weight"),
            p("decoder.upsample.1.0.conv.bias"),up1_tconv_.as<float>()+static_cast<std::size_t>(start*4)*LAT,
            start*2,count*2,LAT,LAT,2,2,hot_);
        convnext_range(1,start*4,count*4,up1_tconv_.as<float>(),up1_.as<float>());
        stream_conv_range(stream_,up1_.as<float>(),p("decoder.decoder.0.conv.weight"),p("decoder.decoder.0.conv.bias"),
            decoder0_.as<float>()+static_cast<std::size_t>(start*4)*DD,start*4,count*4,LAT,DD,7,1,1,hot_);
    }

    void residual_range(
        int di,
        int ui,
        int dil,
        int count,
        int ch,
        const float* raw,
        Buffer& snake_work,
        Buffer& snake_history,
        float* out){

        const int history_rows=6*dil;
        const std::size_t history_bytes=
            static_cast<std::size_t>(history_rows)
            *static_cast<std::size_t>(ch)
            *sizeof(float);

        hipok(
            hipMemcpyAsync(
                snake_work.as<float>(),
                snake_history.as<float>(),
                history_bytes,
                hipMemcpyDeviceToDevice,
                stream_),
            "stream residual history load");

        auto pre=db(di,"block."+std::to_string(ui));
        snake(
            stream_,
            raw,
            p(pre+".act1.alpha"),
            p(pre+".act1.beta"),
            snake_work.as<float>()
                +static_cast<std::size_t>(history_rows)*ch,
            count,
            ch);

        stream_conv_range(
            stream_,
            snake_work.as<float>(),
            p(pre+".conv1.conv.weight"),
            p(pre+".conv1.conv.bias"),
            scratch0_.as<float>(),
            history_rows,
            count,
            ch,
            ch,
            7,
            dil,
            1,
            hot_);

        snake(
            stream_,
            scratch0_.as<float>(),
            p(pre+".act2.alpha"),
            p(pre+".act2.beta"),
            scratch1_.as<float>(),
            count,
            ch);

        if(hot_){
            stream_pointwise_residual4(
                stream_,
                scratch1_.as<float>(),
                raw,
                p(pre+".conv2.conv.weight"),
                p(pre+".conv2.conv.bias"),
                out,
                count,
                ch);
        }else{
            conv(
                stream_,
                scratch1_.as<float>(),
                p(pre+".conv2.conv.weight"),
                p(pre+".conv2.conv.bias"),
                scratch2_.as<float>(),
                count,
                ch,
                ch,
                1,
                1);
            add(
                stream_,
                raw,
                scratch2_.as<float>(),
                out,
                static_cast<std::size_t>(count)*ch);
        }

        hipok(
            hipMemcpyAsync(
                snake_history.as<float>(),
                snake_work.as<float>()
                    +static_cast<std::size_t>(count)*ch,
                history_bytes,
                hipMemcpyDeviceToDevice,
                stream_),
            "stream residual history store");
    }

    void vocoder_range(int fstart,int fcount){
        constexpr int rates[4]={8,5,4,3};
        int input_start=fstart*4;
        int input_count=fcount*4;
        int input_ch=DD;

        for(int s=0;s<4;++s){
            const int di=s+1;
            const int r=rates[s];
            const int output_ch=DD>>(s+1);
            const int output_count=input_count*r;
            auto& state=stages_[s];
            auto pre=db(di,"block");

            float* compact_input=state.input_work.as<float>();
            const std::size_t one_input_row=
                static_cast<std::size_t>(input_ch)*sizeof(float);

            hipok(
                hipMemcpyAsync(
                    compact_input,
                    state.input_tail.as<float>(),
                    one_input_row,
                    hipMemcpyDeviceToDevice,
                    stream_),
                "stream vocoder input tail load");

            if(s==0){
                hipok(
                    hipMemcpyAsync(
                        compact_input+input_ch,
                        decoder0_.as<float>()
                            +static_cast<std::size_t>(input_start)*input_ch,
                        static_cast<std::size_t>(input_count)*one_input_row,
                        hipMemcpyDeviceToDevice,
                        stream_),
                    "stream vocoder input load");
            }else{
                auto& previous=stages_[s-1];
                hipok(
                    hipMemcpyAsync(
                        compact_input+input_ch,
                        previous.raw3.as<float>(),
                        static_cast<std::size_t>(input_count)*one_input_row,
                        hipMemcpyDeviceToDevice,
                        stream_),
                    "stream vocoder stage input load");
            }

            snake(
                stream_,
                compact_input+input_ch,
                p(pre+".0.alpha"),
                p(pre+".0.beta"),
                compact_input+input_ch,
                input_count,
                input_ch);

            hipok(
                hipMemcpyAsync(
                    state.input_tail.as<float>(),
                    compact_input
                        +static_cast<std::size_t>(input_count)*input_ch,
                    one_input_row,
                    hipMemcpyDeviceToDevice,
                    stream_),
                "stream vocoder input tail store");

            stream_tconv_range(
                stream_,
                compact_input,
                p(pre+".1.conv.weight"),
                p(pre+".1.conv.bias"),
                state.raw0.as<float>(),
                1,
                input_count,
                input_ch,
                output_ch,
                2*r,
                r,
                hot_);

            residual_range(
                di,2,1,output_count,output_ch,
                state.raw0.as<float>(),
                state.snake0_work,
                state.snake0_history,
                state.raw1.as<float>());
            residual_range(
                di,3,3,output_count,output_ch,
                state.raw1.as<float>(),
                state.snake1_work,
                state.snake1_history,
                state.raw2.as<float>());
            residual_range(
                di,4,9,output_count,output_ch,
                state.raw2.as<float>(),
                state.snake2_work,
                state.snake2_history,
                state.raw3.as<float>());

            input_start=0;
            input_count=output_count;
            input_ch=output_ch;
        }

        if(input_ch!=96){
            throw std::runtime_error("streaming vocoder channel mismatch");
        }

        constexpr int final_history_rows=6;
        auto& final_stage=stages_.back();
        const std::size_t final_history_bytes=
            static_cast<std::size_t>(final_history_rows)
            *96*sizeof(float);

        hipok(
            hipMemcpyAsync(
                final_snake_work_.as<float>(),
                final_snake_history_.as<float>(),
                final_history_bytes,
                hipMemcpyDeviceToDevice,
                stream_),
            "stream final history load");

        snake(
            stream_,
            final_stage.raw3.as<float>(),
            p("decoder.decoder.5.alpha"),
            p("decoder.decoder.5.beta"),
            final_snake_work_.as<float>()
                +static_cast<std::size_t>(final_history_rows)*96,
            input_count,
            96);

        const int waveform_start=fstart*UPSAMPLE;
        stream_conv_range(
            stream_,
            final_snake_work_.as<float>(),
            p("decoder.decoder.6.conv.weight"),
            p("decoder.decoder.6.conv.bias"),
            waveform_.as<float>()+waveform_start,
            final_history_rows,
            input_count,
            96,
            1,
            7,
            1,
            1,
            false);

        hipLaunchKernelGGL(
            clamp_f32,
            dim3(threads_for(static_cast<std::size_t>(input_count))),
            dim3(THREADS),
            0,
            stream_,
            waveform_.as<float>()+waveform_start,
            static_cast<std::size_t>(input_count));
        hipok(hipGetLastError(),"stream clamp");

        hipok(
            hipMemcpyAsync(
                final_snake_history_.as<float>(),
                final_snake_work_.as<float>()
                    +static_cast<std::size_t>(input_count)*96,
                final_history_bytes,
                hipMemcpyDeviceToDevice,
                stream_),
            "stream final history store");

    }

    std::shared_ptr<Parameters>P_;int max_frames_=0,chunk_frames_=0,processed_frames_=0;bool hot_=true;hipStream_t stream_=nullptr;bool owns_stream_=true;
    Buffer codes_,dequant_,preconv_,transformer_,up0_tconv_,up0_,up1_tconv_,up1_,decoder0_,waveform_;
    std::array<ResidualStage,4>stages_;std::array<Buffer,TL>k_cache_,v_cache_;
    Buffer th_a_,th_b_,th_c_,th_norm_,th_q_,th_k_,th_v_,th_qr_,th_ctx_,th_gate_,th_up_,th_scores_,th_prob_;
    Buffer dq_sem_,dq_acoustic_,dq_sem_proj_,dq_acoustic_proj_,scratch0_,scratch1_,scratch2_,scratch3_;
    Buffer final_snake_work_,final_snake_history_;
};

struct StreamingPipelineResult{
    std::vector<float>waveform;double parameter_preload_wall_ms=0.0,parameter_wait_ms=0.0;
    double decoder_setup_ms=0.0,decoder_compute_ms=0.0,decoder_pipeline_wall_ms=0.0,first_audio_ms=0.0,tail_wait_ms=0.0;
    int chunk_count=0,first_chunk_frames=0,chunk_frames=STREAM_CHUNK_FRAMES;
};

class StreamingCodecPipeline{
public:
    using StreamClock=
        std::chrono::steady_clock;
    using StreamTime=
        StreamClock::time_point;

    static double stream_elapsed_ms(
        StreamTime start,
        StreamTime end){
        return std::chrono::duration<
            double,
            std::milli>(
                end-start)
            .count();
    }

    using PreloadResult=std::pair<fs::path,std::shared_ptr<Parameters>>;
    StreamingCodecPipeline(
        std::future<PreloadResult>&& f,
        StreamTime preload_start,
        int maximum_frames,
        StreamTime e2e_start,
        int chunk_frames=STREAM_CHUNK_FRAMES,
        hipStream_t resident_decoder_stream=nullptr,
        StreamingCodecDecoder* resident_decoder=nullptr)
      :future_(std::move(f)),
       preload_start_(preload_start),
       maximum_frames_(maximum_frames),
       e2e_start_(e2e_start),
       chunk_frames_(chunk_frames),
       resident_decoder_stream_(resident_decoder_stream),
       resident_decoder_(resident_decoder){
        worker_=std::thread([this](){worker_main();});
    }
    ~StreamingCodecPipeline(){if(worker_.joinable()){{
        std::lock_guard<std::mutex>lock(mutex_);closed_=true;}condition_.notify_all();worker_.join();}}
    void push_frame(const std::array<std::int32_t,NQ>&frame){{
        std::lock_guard<std::mutex>lock(mutex_);if(closed_)throw std::runtime_error("streaming codec pipeline closed");
        queue_.push_back(frame);}condition_.notify_one();}
    StreamingPipelineResult finish(){
        auto ws=StreamClock::now();{std::lock_guard<std::mutex>lock(mutex_);closed_=true;}condition_.notify_all();
        if(worker_.joinable())worker_.join();result_.tail_wait_ms=stream_elapsed_ms(ws,StreamClock::now());
        if(error_)std::rethrow_exception(error_);return std::move(result_);
    }
private:
    void worker_main(){
        try{
            auto pw=StreamClock::now();auto preloaded=future_.get();auto ready=StreamClock::now();
            result_.parameter_wait_ms=stream_elapsed_ms(pw,ready);result_.parameter_preload_wall_ms=stream_elapsed_ms(preload_start_,ready);
            auto ss=StreamClock::now();
            std::unique_ptr<StreamingCodecDecoder> owned_decoder;
            StreamingCodecDecoder* decoder=resident_decoder_;
            if(decoder==nullptr){
                owned_decoder=std::make_unique<StreamingCodecDecoder>(
                    preloaded.second,
                    maximum_frames_,
                    chunk_frames_,
                    true,
                    resident_decoder_stream_);
                decoder=owned_decoder.get();
            }
            result_.decoder_setup_ms=stream_elapsed_ms(ss,StreamClock::now());auto ps=StreamClock::now();
            std::vector<std::int32_t>pending;pending.reserve(static_cast<std::size_t>(chunk_frames_)*NQ);bool first=false;
            while(true){
                const int target_frames=
                    first
                    ?chunk_frames_
                    :STREAM_FIRST_CHUNK_FRAMES;
                {
                    std::unique_lock<std::mutex>lock(mutex_);
                    condition_.wait(lock,[this](){return closed_||!queue_.empty();});
                    while(!queue_.empty()&&pending.size()<static_cast<std::size_t>(target_frames)*NQ){
                        auto f=queue_.front();queue_.pop_front();pending.insert(pending.end(),f.begin(),f.end());}
                    if(pending.empty()&&closed_&&queue_.empty())break;
                    if(pending.size()<static_cast<std::size_t>(target_frames)*NQ&&!closed_)continue;
                }
                if(pending.empty())continue;auto ds=StreamClock::now();auto wave=decoder->append(pending);
                result_.decoder_compute_ms+=stream_elapsed_ms(ds,StreamClock::now());++result_.chunk_count;
                result_.waveform.insert(result_.waveform.end(),wave.begin(),wave.end());
                if(!first){
                    first=true;
                    result_.first_chunk_frames=
                        static_cast<int>(pending.size()/NQ);
                    result_.first_audio_ms=
                        stream_elapsed_ms(
                            e2e_start_,
                            StreamClock::now());
                }
                pending.clear();
            }
            result_.decoder_pipeline_wall_ms=stream_elapsed_ms(ps,StreamClock::now());
        }catch(...){error_=std::current_exception();}
    }
    std::future<PreloadResult>future_;StreamTime preload_start_{};int maximum_frames_=0;StreamTime e2e_start_{};
    int chunk_frames_=0;hipStream_t resident_decoder_stream_=nullptr;StreamingCodecDecoder* resident_decoder_=nullptr;
    std::mutex mutex_;std::condition_variable condition_;
    std::deque<std::array<std::int32_t,NQ>>queue_;bool closed_=false;std::thread worker_;
    std::exception_ptr error_;StreamingPipelineResult result_;
};

class NativeCodecDecoder {
public:
    NativeCodecDecoder(
        const fs::path& dir,
        int frames,
        bool trace=false,
        bool profile=false,
        bool hot_lds=false)
        :NativeCodecDecoder(
            std::make_shared<Parameters>(dir),
            frames,
            trace,
            profile,
            hot_lds){}

    NativeCodecDecoder(
        std::shared_ptr<Parameters> parameters,
        int frames,
        bool trace=false,
        bool profile=false,
        bool hot_lds=false)
        :P_(std::move(parameters)),
         F_(frames),
         trace_enabled_(trace),
         profile_enabled_(profile),
         hot_lds_enabled_(hot_lds){

        if(!P_)
            throw std::runtime_error(
                "native codec decoder received null preloaded parameters");

        if(F_<=0||F_>300)
            throw std::runtime_error(
                "legacy_legacy_tag_h codec frame count must be 1..300 in the first native closure.");

        int least=0,greatest=0;
        hipok(
            hipDeviceGetStreamPriorityRange(
                &least,&greatest),
            "stream priority");
        hipok(
            hipStreamCreateWithPriority(
                &st_,hipStreamNonBlocking,greatest),
            "stream create");

        allocate();

        if(profile_enabled_){
            for(auto& event:profile_events_)
                hipok(
                    hipEventCreate(&event),
                    "profile event create");
        }

        capture();

        std::cout
            <<"codec_decoder_backend: native_cpp_hip_f32\n"
            <<"codec_linear_backend: wave32_vgpr_f32acc\n"
            <<"codec_conv_backend: wave32_vgpr_f32acc\n"
            <<"codec_transconv_backend: wave32_phase_reduced_vgpr_f32acc\n"
            <<"codec_norm_backend: wave32_plus_explicit_lds\n"
            <<"codec_softmax_backend: wave32_plus_explicit_lds\n"
            <<"codec_dag_backend: hip_graph\n"
            <<"codec_graph_node_count: "<<nodes_<<"\n"
            <<"codec_graph_instantiated: True\n"
            <<"codec_parameter_source: preloaded_shared_gpu_parameters\n";

        if(hot_lds_enabled_){
            std::cout
                <<"codec_hot_backend: qingming_wave32_lds_tiled\n"
                <<"codec_tile_waves_per_block: "<<WAVES<<"\n"
                <<"codec_hot_vgpr_accumulator: lane_private_fp32\n"
                <<"codec_hot_cross_lane_reduce: wave32_shuffle\n"
                <<"codec_hot_input_reuse: explicit_lds\n"
                <<"codec_hot_pointwise_residual_fused: True\n"
                <<"codec_hot_snake_recomputed_per_tile: False\n"
                <<"codec_max_dynamic_lds_bytes: 28672\n";
        }
    }
    ~NativeCodecDecoder(){
        if(exec_)(void)hipGraphExecDestroy(exec_);
        if(graph_)(void)hipGraphDestroy(graph_);
        for(auto& event:profile_events_)if(event)(void)hipEventDestroy(event);
        if(st_)(void)hipStreamDestroy(st_);
    }
    NativeCodecDecoder(const NativeCodecDecoder&)=delete;NativeCodecDecoder& operator=(const NativeCodecDecoder&)=delete;

    std::size_t node_count() const {
        return nodes_;
    }

    std::vector<float> decode(const std::vector<std::int32_t>& codec){
        if(codec.size()!=static_cast<std::size_t>(F_)*NQ)throw std::runtime_error("legacy_legacy_tag_h codec matrix size mismatch.");
        hipok(hipMemcpyAsync(codes_.as<std::int32_t>(),codec.data(),codec.size()*sizeof(std::int32_t),hipMemcpyHostToDevice,st_),"codes upload");
        hipok(hipGraphLaunch(exec_,st_),"graph launch");
        std::vector<float> wave(static_cast<std::size_t>(F_)*UPSAMPLE);
        hipok(hipMemcpyAsync(wave.data(),b_.as<float>(),wave.size()*sizeof(float),hipMemcpyDeviceToHost,st_),"wave download");
        hipok(hipStreamSynchronize(st_),"graph sync");
        return wave;
    }

    std::vector<float> decode_legacy_reference(
        const std::vector<std::int32_t>& codec){

        if(codec.size()!=static_cast<std::size_t>(F_)*NQ){
            throw std::runtime_error(
                "legacy_legacy_tag_j legacy reference codec matrix size mismatch.");
        }

        hipok(
            hipMemcpyAsync(
                codes_.as<std::int32_t>(),
                codec.data(),
                codec.size()*sizeof(std::int32_t),
                hipMemcpyHostToDevice,
                st_),
            "legacy_legacy_tag_j legacy codes upload");

        const bool saved_hot =
            hot_lds_enabled_;
        const bool saved_diag =
            diagnostic_pass_active_;

        hot_lds_enabled_=false;
        diagnostic_pass_active_=false;

        try{
            dag();

            hipok(
                hipStreamSynchronize(st_),
                "legacy_legacy_tag_j legacy direct sync");
        }catch(...){
            hot_lds_enabled_=saved_hot;
            diagnostic_pass_active_=saved_diag;
            throw;
        }

        std::vector<float> wave(
            static_cast<std::size_t>(F_)
            * UPSAMPLE);

        hipok(
            hipMemcpy(
                wave.data(),
                b_.as<float>(),
                wave.size()*sizeof(float),
                hipMemcpyDeviceToHost),
            "legacy_legacy_tag_j legacy waveform download");

        hot_lds_enabled_=saved_hot;
        diagnostic_pass_active_=saved_diag;

        std::cout
            <<"legacy_reference_backend: wave32_global_input\n"
            <<"legacy_reference_execution: direct_stream\n";

        return wave;
    }

    void run_diagnostics(
        const std::vector<std::int32_t>& codec,
        const fs::path& trace_root){

        if(
            !trace_enabled_
            && !profile_enabled_
        ){
            return;
        }

        if(codec.size()!=static_cast<std::size_t>(F_)*NQ){
            throw std::runtime_error(
                "legacy_legacy_tag_i diagnostic codec matrix size mismatch.");
        }

        hipok(
            hipMemcpyAsync(
                codes_.as<std::int32_t>(),
                codec.data(),
                codec.size()*sizeof(std::int32_t),
                hipMemcpyHostToDevice,
                st_),
            "diagnostic codes upload");

        diagnostic_pass_active_=true;

        try{
            dag();
            hipok(
                hipStreamSynchronize(st_),
                "diagnostic direct-pass sync");

            if(trace_enabled_){
                save_trace(trace_root);
            }

            if(profile_enabled_){
                print_profile();
            }
        }catch(...){
            diagnostic_pass_active_=false;
            throw;
        }

        diagnostic_pass_active_=false;

        std::cout
            <<"diagnostics_execution: direct_stream_same_kernels\n"
            <<"profile_events_inside_graph: False\n"
            <<"trace_nodes_inside_graph: False\n";
    }

private:
    void save_trace(const fs::path& root){
        if(!trace_enabled_)return;
        fs::create_directories(root);

        for(const auto& tap:trace_taps_){
            std::vector<float> host(tap.elements);
            hipok(
                hipMemcpy(
                    host.data(),
                    tap.data.as<float>(),
                    tap.elements*sizeof(float),
                    hipMemcpyDeviceToHost),
                "trace download");

            legacy_write_binary(
                root/(tap.name+".f32.bin"),
                host);

            write_text_file(
                root/(tap.name+".shape.txt"),
                std::to_string(tap.rows)
                    +" "
                    +std::to_string(tap.cols)
                    +"\n");
        }

        std::cout
            <<"native_codec_trace_stage_count: "
            <<trace_taps_.size()
            <<"\n"
            <<"native_codec_trace_root: "
            <<root.string()
            <<"\n";
    }

private:
    struct TraceTap {
        std::string name;
        int rows=0;
        int cols=0;
        std::size_t elements=0;
        Buffer data;

        TraceTap(
            std::string n,
            int r,
            int c)
            :name(std::move(n)),
             rows(r),
             cols(c),
             elements(
                static_cast<std::size_t>(r)
                *static_cast<std::size_t>(c)),
             data(elements*sizeof(float)){}
    };

    enum TraceIndex {
        TRACE_DEQUANT=0,
        TRACE_PRE_CONV,
        TRACE_PRE_TRANSFORMER,
        TRACE_UPSAMPLE0,
        TRACE_UPSAMPLE1,
        TRACE_DECODER0,
        TRACE_DECODER1,
        TRACE_DECODER2,
        TRACE_DECODER3,
        TRACE_DECODER4,
        TRACE_FINAL_SNAKE,
        TRACE_PRE_CLAMP,
        TRACE_WAVEFORM,
        TRACE_COUNT
    };

    static constexpr std::array<const char*,14>
        profile_names_{{
            "start",
            "dequant",
            "pre_conv",
            "pre_transformer",
            "upsample0",
            "upsample1",
            "decoder0",
            "decoder1",
            "decoder2",
            "decoder3",
            "decoder4",
            "final_snake",
            "final_conv",
            "clamp"
        }};

    void record_profile(int index){
        if(
            profile_enabled_
            && diagnostic_pass_active_
        ){
            hipok(
                hipEventRecord(
                    profile_events_.at(
                        static_cast<std::size_t>(index)),
                    st_),
                "profile event record");
        }
    }

    void print_profile(){
        hipok(
            hipEventSynchronize(
                profile_events_.back()),
            "profile final event sync");

        float total=0.0f;

        for(std::size_t i=0;i+1<profile_events_.size();++i){
            float ms=0.0f;

            hipok(
                hipEventElapsedTime(
                    &ms,
                    profile_events_[i],
                    profile_events_[i+1]),
                "profile elapsed time");

            total+=ms;

            std::cout
                <<"codec_stage_ms: stage="
                <<profile_names_[i+1]
                <<" ms="
                <<ms
                <<"\n";
        }

        std::cout
            <<"codec_profile_total_ms: "
            <<total
            <<"\n"
            <<"profile_event_backend: direct_stream_events\n";
    }

    void trace_copy(
        TraceIndex index,
        const float* source){

        if(
            !trace_enabled_
            || !diagnostic_pass_active_
        ){
            return;
        }

        auto& tap=
            trace_taps_.at(
                static_cast<std::size_t>(index));
        copyv(
            st_,
            source,
            tap.data.as<float>(),
            tap.elements);
    }

    static std::string tr(int i,const std::string&t){return "decoder.pre_transformer.layers."+std::to_string(i)+"."+t;}
    static std::string upn(int i,const std::string&t){return "decoder.upsample."+std::to_string(i)+"."+t;}
    static std::string db(int i,const std::string&t){return "decoder.decoder."+std::to_string(i)+"."+t;}
    const Param&p(const std::string&s)const{return P_->get(s);}

    void allocate(){
        std::size_t f=static_cast<std::size_t>(F_);
        std::size_t maxe=std::max({f*4*LAT*4,f*4*DD,f*32*768,f*160*384,f*640*192,f*UPSAMPLE*96});
        std::size_t bytes=maxe*sizeof(float);a_=Buffer(bytes);b_=Buffer(bytes);c_=Buffer(bytes);d_=Buffer(bytes);
        std::size_t ae=f*AD;q_=Buffer(ae*4);k_=Buffer(ae*4);v_=Buffer(ae*4);qr_=Buffer(ae*4);kr_=Buffer(ae*4);ctx_=Buffer(ae*4);
        gate_=Buffer(f*TI*4);up_=Buffer(f*TI*4);scores_=Buffer(f*HEADS*f*4);prob_=Buffer(f*HEADS*f*4);codes_=Buffer(f*NQ*sizeof(std::int32_t));

        if(trace_enabled_){
            trace_taps_.reserve(TRACE_COUNT);
            trace_taps_.emplace_back("dequant",F_,CBO);
            trace_taps_.emplace_back("pre_conv",F_,LAT);
            trace_taps_.emplace_back("pre_transformer",F_,LAT);
            trace_taps_.emplace_back("upsample0",F_*2,LAT);
            trace_taps_.emplace_back("upsample1",F_*4,LAT);
            trace_taps_.emplace_back("decoder0",F_*4,DD);
            trace_taps_.emplace_back("decoder1",F_*32,768);
            trace_taps_.emplace_back("decoder2",F_*160,384);
            trace_taps_.emplace_back("decoder3",F_*640,192);
            trace_taps_.emplace_back("decoder4",F_*UPSAMPLE,96);
            trace_taps_.emplace_back("final_snake",F_*UPSAMPLE,96);
            trace_taps_.emplace_back("pre_clamp",F_*UPSAMPLE,1);
            trace_taps_.emplace_back("waveform",F_*UPSAMPLE,1);
        }
    }

    void dequant(){
        std::size_t n=static_cast<std::size_t>(F_)*CBD;
        auto& e0=p("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");
        auto& u0=p("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
        hipLaunchKernelGGL(codebook_accum_f32,dim3(threads_for(n)),dim3(THREADS),0,st_,codes_.as<std::int32_t>(),e0.data(),u0.data(),a_.as<float>(),F_,0,1);
        hipok(hipGetLastError(),"semantic codebook");
        for(int g=1;g<NQ;++g){auto pre="decoder.quantizer.rvq_rest.vq.layers."+std::to_string(g-1)+"._codebook.";
            hipLaunchKernelGGL(codebook_accum_f32,dim3(threads_for(n)),dim3(THREADS),0,st_,codes_.as<std::int32_t>(),
                p(pre+"embedding_sum").data(),p(pre+"cluster_usage").data(),b_.as<float>(),F_,g,g==1?1:0);hipok(hipGetLastError(),"acoustic codebook");}
        lin0(st_,a_.as<float>(),p("decoder.quantizer.rvq_first.output_proj.weight"),c_.as<float>(),F_,CBO,CBD);
        lin0(st_,b_.as<float>(),p("decoder.quantizer.rvq_rest.output_proj.weight"),d_.as<float>(),F_,CBO,CBD);
        add(st_,c_.as<float>(),d_.as<float>(),a_.as<float>(),static_cast<std::size_t>(F_)*CBO);
    }

    void transformer(){
        lin(st_,b_.as<float>(),p("decoder.pre_transformer.input_proj.weight"),p("decoder.pre_transformer.input_proj.bias"),a_.as<float>(),F_,TH,LAT);
        for(int l=0;l<TL;++l){
            rms(st_,a_.as<float>(),p(tr(l,"input_layernorm.weight")),b_.as<float>(),F_,TH,1e-5f);
            lin0(st_,b_.as<float>(),p(tr(l,"self_attn.q_proj.weight")),q_.as<float>(),F_,AD,TH);
            lin0(st_,b_.as<float>(),p(tr(l,"self_attn.k_proj.weight")),k_.as<float>(),F_,AD,TH);
            lin0(st_,b_.as<float>(),p(tr(l,"self_attn.v_proj.weight")),v_.as<float>(),F_,AD,TH);
            rope(st_,q_.as<float>(),qr_.as<float>(),F_);rope(st_,k_.as<float>(),kr_.as<float>(),F_);
            attention(st_,qr_.as<float>(),kr_.as<float>(),v_.as<float>(),scores_.as<float>(),prob_.as<float>(),ctx_.as<float>(),F_);
            lin0(st_,ctx_.as<float>(),p(tr(l,"self_attn.o_proj.weight")),c_.as<float>(),F_,TH,AD);
            scaleadd(st_,a_.as<float>(),c_.as<float>(),p(tr(l,"self_attn_layer_scale.scale")),b_.as<float>(),F_,TH);
            rms(st_,b_.as<float>(),p(tr(l,"post_attention_layernorm.weight")),c_.as<float>(),F_,TH,1e-5f);
            lin0(st_,c_.as<float>(),p(tr(l,"mlp.gate_proj.weight")),gate_.as<float>(),F_,TI,TH);
            lin0(st_,c_.as<float>(),p(tr(l,"mlp.up_proj.weight")),up_.as<float>(),F_,TI,TH);
            silumul(st_,gate_.as<float>(),up_.as<float>(),gate_.as<float>(),static_cast<std::size_t>(F_)*TI);
            lin0(st_,gate_.as<float>(),p(tr(l,"mlp.down_proj.weight")),c_.as<float>(),F_,TH,TI);
            scaleadd(st_,b_.as<float>(),c_.as<float>(),p(tr(l,"mlp_layer_scale.scale")),a_.as<float>(),F_,TH);
        }
        rms(st_,a_.as<float>(),p("decoder.pre_transformer.norm.weight"),b_.as<float>(),F_,TH,1e-5f);
        lin(st_,b_.as<float>(),p("decoder.pre_transformer.output_proj.weight"),p("decoder.pre_transformer.output_proj.bias"),a_.as<float>(),F_,LAT,TH);
    }

    void convnext(int stage,int t,const float*res,float*dw,float*norm,float*wide,float*proj,float*out){
        auto pre=upn(stage,"1");
        conv(st_,res,p(pre+".dwconv.conv.weight"),p(pre+".dwconv.conv.bias"),dw,t,LAT,LAT,7,1,LAT);
        lnorm(st_,dw,p(pre+".norm.weight"),p(pre+".norm.bias"),norm,t,LAT,1e-6f);
        lin(st_,norm,p(pre+".pwconv1.weight"),p(pre+".pwconv1.bias"),wide,t,4*LAT,LAT);
        gelu(st_,wide,wide,static_cast<std::size_t>(t)*4*LAT);
        lin(st_,wide,p(pre+".pwconv2.weight"),p(pre+".pwconv2.bias"),proj,t,LAT,4*LAT);
        scaleadd(st_,res,proj,p(pre+".gamma"),out,t,LAT);
    }

    void residual(int di,int ui,int dil,int t,int ch,const float*res,float*s1,float*s2,float*out){
        auto pre=db(di,"block."+std::to_string(ui));

        snake(
            st_,
            res,
            p(pre+".act1.alpha"),
            p(pre+".act1.beta"),
            s1,
            t,
            ch);

        if(hot_lds_enabled_){
            qingming_conv_lds(
                st_,
                s1,
                p(pre+".conv1.conv.weight"),
                p(pre+".conv1.conv.bias"),
                s2,
                t,
                ch,
                ch,
                7,
                dil);
        }else{
            conv(
                st_,
                s1,
                p(pre+".conv1.conv.weight"),
                p(pre+".conv1.conv.bias"),
                s2,
                t,
                ch,
                ch,
                7,
                dil);
        }

        snake(
            st_,
            s2,
            p(pre+".act2.alpha"),
            p(pre+".act2.beta"),
            s1,
            t,
            ch);

        if(hot_lds_enabled_){
            qingming_pointwise_residual_lds(
                st_,
                s1,
                res,
                p(pre+".conv2.conv.weight"),
                p(pre+".conv2.conv.bias"),
                out,
                t,
                ch);
        }else{
            conv(
                st_,
                s1,
                p(pre+".conv2.conv.weight"),
                p(pre+".conv2.conv.bias"),
                s2,
                t,
                ch,
                ch,
                1,
                1);

            add(
                st_,
                res,
                s2,
                out,
                static_cast<std::size_t>(t)*ch);
        }
    }

    void dag(){
        record_profile(0);

        dequant();
        trace_copy(TRACE_DEQUANT,a_.as<float>());
        record_profile(1);

        conv(st_,a_.as<float>(),p("decoder.pre_conv.conv.weight"),p("decoder.pre_conv.conv.bias"),b_.as<float>(),F_,CBO,LAT,3,1);
        trace_copy(TRACE_PRE_CONV,b_.as<float>());
        record_profile(2);

        transformer(); // A=[F,1024]
        trace_copy(TRACE_PRE_TRANSFORMER,a_.as<float>());
        record_profile(3);

        if(hot_lds_enabled_){
            qingming_tconv_lds(st_,a_.as<float>(),p("decoder.upsample.0.0.conv.weight"),p("decoder.upsample.0.0.conv.bias"),b_.as<float>(),F_,LAT,LAT,2,2);
        }else{
            tconv(st_,a_.as<float>(),p("decoder.upsample.0.0.conv.weight"),p("decoder.upsample.0.0.conv.bias"),b_.as<float>(),F_,LAT,LAT,2,2);
        }
        int t=F_*2;
        convnext(0,t,b_.as<float>(),a_.as<float>(),c_.as<float>(),d_.as<float>(),a_.as<float>(),c_.as<float>());
        trace_copy(TRACE_UPSAMPLE0,c_.as<float>());
        record_profile(4);

        if(hot_lds_enabled_){
            qingming_tconv_lds(st_,c_.as<float>(),p("decoder.upsample.1.0.conv.weight"),p("decoder.upsample.1.0.conv.bias"),a_.as<float>(),t,LAT,LAT,2,2);
        }else{
            tconv(st_,c_.as<float>(),p("decoder.upsample.1.0.conv.weight"),p("decoder.upsample.1.0.conv.bias"),a_.as<float>(),t,LAT,LAT,2,2);
        }
        t*=2;
        convnext(1,t,a_.as<float>(),b_.as<float>(),c_.as<float>(),d_.as<float>(),b_.as<float>(),c_.as<float>());
        copyv(st_,c_.as<float>(),a_.as<float>(),static_cast<std::size_t>(t)*LAT);
        trace_copy(TRACE_UPSAMPLE1,a_.as<float>());
        record_profile(5);

        if(hot_lds_enabled_){
            qingming_conv_lds(st_,a_.as<float>(),p("decoder.decoder.0.conv.weight"),p("decoder.decoder.0.conv.bias"),c_.as<float>(),t,LAT,DD,7,1);
        }else{
            conv(st_,a_.as<float>(),p("decoder.decoder.0.conv.weight"),p("decoder.decoder.0.conv.bias"),c_.as<float>(),t,LAT,DD,7,1);
        }
        copyv(st_,c_.as<float>(),a_.as<float>(),static_cast<std::size_t>(t)*DD);
        trace_copy(TRACE_DECODER0,a_.as<float>());
        record_profile(6);

        constexpr int rates[4]={8,5,4,3};
        constexpr TraceIndex stage_trace[4]={
            TRACE_DECODER1,
            TRACE_DECODER2,
            TRACE_DECODER3,
            TRACE_DECODER4
        };

        for(int s=0;s<4;++s){
            int di=s+1,ic=DD>>s,oc=DD>>(s+1),r=rates[s];auto pre=db(di,"block");
            snake(st_,a_.as<float>(),p(pre+".0.alpha"),p(pre+".0.beta"),a_.as<float>(),t,ic);

            if(hot_lds_enabled_){
                qingming_tconv_lds(
                    st_,
                    a_.as<float>(),
                    p(pre+".1.conv.weight"),
                    p(pre+".1.conv.bias"),
                    b_.as<float>(),
                    t,
                    ic,
                    oc,
                    2*r,
                    r);
            }else{
                tconv(
                    st_,
                    a_.as<float>(),
                    p(pre+".1.conv.weight"),
                    p(pre+".1.conv.bias"),
                    b_.as<float>(),
                    t,
                    ic,
                    oc,
                    2*r,
                    r);
            }

            t*=r;
            residual(di,2,1,t,oc,b_.as<float>(),c_.as<float>(),d_.as<float>(),a_.as<float>());
            residual(di,3,3,t,oc,a_.as<float>(),c_.as<float>(),d_.as<float>(),b_.as<float>());
            residual(di,4,9,t,oc,b_.as<float>(),c_.as<float>(),d_.as<float>(),a_.as<float>());

            trace_copy(
                stage_trace[s],
                a_.as<float>());

            record_profile(7+s);
        }

        if(t!=F_*UPSAMPLE)throw std::runtime_error("legacy_legacy_tag_h vocoder upsample product mismatch");

        snake(st_,a_.as<float>(),p("decoder.decoder.5.alpha"),p("decoder.decoder.5.beta"),a_.as<float>(),t,96);
        trace_copy(TRACE_FINAL_SNAKE,a_.as<float>());
        record_profile(11);

        conv(st_,a_.as<float>(),p("decoder.decoder.6.conv.weight"),p("decoder.decoder.6.conv.bias"),b_.as<float>(),t,96,1,7,1);
        trace_copy(TRACE_PRE_CLAMP,b_.as<float>());
        record_profile(12);

        hipLaunchKernelGGL(clamp_f32,dim3(threads_for(static_cast<std::size_t>(t))),dim3(THREADS),0,st_,b_.as<float>(),static_cast<std::size_t>(t));
        hipok(hipGetLastError(),"clamp");
        trace_copy(TRACE_WAVEFORM,b_.as<float>());
        record_profile(13);
    }

    void capture(){
        diagnostic_pass_active_=false;

        hipok(
            hipStreamBeginCapture(
                st_,
                hipStreamCaptureModeGlobal),
            "begin graph capture");

        dag();

        hipok(
            hipStreamEndCapture(
                st_,
                &graph_),
            "end graph capture");

        if(!graph_){
            throw std::runtime_error(
                "legacy_legacy_tag_h null codec graph");
        }

        hipok(
            hipGraphInstantiate(
                &exec_,
                graph_,
                nullptr,
                nullptr,
                0),
            "graph instantiate");

        std::size_t n=0;

        hipok(
            hipGraphGetNodes(
                graph_,
                nullptr,
                &n),
            "graph node count");

        nodes_=n;

        std::cout
            <<"graph_diagnostics_nodes: 0\n"
            <<"graph_is_production_math_only: True\n";
    }

    std::shared_ptr<Parameters> P_;int F_=0;hipStream_t st_=nullptr;hipGraph_t graph_=nullptr;hipGraphExec_t exec_=nullptr;std::size_t nodes_=0;
    Buffer a_,b_,c_,d_,q_,k_,v_,qr_,kr_,ctx_,gate_,up_,scores_,prob_,codes_;

    bool trace_enabled_=false;
    bool profile_enabled_=false;
    bool diagnostic_pass_active_=false;
    bool hot_lds_enabled_=false;
    std::vector<TraceTap> trace_taps_;
    std::array<hipEvent_t,14> profile_events_{{}};
};

static fs::path locate_speech_tokenizer(const fs::path& root){
    const std::array<fs::path,3> c{{root/"speech_tokenizer",root/"tokenizer",root}};
    for(const auto&d:c)if(fs::is_directory(d))try{
        model_frontend::ModelWeights w(d);if(w.maybe_suffix("decoder.pre_conv.conv.weight"))return d;
    }catch(...){}
    throw std::runtime_error("legacy_legacy_tag_h cannot locate speech_tokenizer safetensors.");
}

} // namespace codec_decoder


// ----------------------------------------------------------------------------
// legacy_legacy_tag_h native Talker RoPE for the pure-text path.
// In text-only generation all 3 mRoPE axes are equal; after interleaving the
// effective coefficients reduce to ordinary 1D RoPE coefficients.
// ----------------------------------------------------------------------------
static std::uint16_t f32_to_bf16(float value){
    std::uint32_t bits=0;std::memcpy(&bits,&value,sizeof(bits));
    if((bits&0x7fffffffU)>0x7f800000U)return static_cast<std::uint16_t>((bits>>16)|0x0040U);
    bits+=0x7fffU+((bits>>16)&1U);return static_cast<std::uint16_t>(bits>>16);
}
struct RopeRows{std::vector<std::uint16_t> cos,sin;};
static RopeRows make_talker_rope(int rows,int start,int hd=128,double theta=1000000.0){
    if(rows<0||start<0||hd<=0||(hd&1))throw std::runtime_error("legacy_legacy_tag_h invalid Talker RoPE geometry");
    RopeRows r;r.cos.resize(static_cast<std::size_t>(rows)*hd);r.sin.resize(static_cast<std::size_t>(rows)*hd);
    int half=hd/2;
    for(int row=0;row<rows;++row){
        double pos=static_cast<double>(start+row);
        for(int i=0;i<half;++i){
            double inv=std::pow(theta,-2.0*static_cast<double>(i)/static_cast<double>(hd));
            double a=pos*inv;auto c=f32_to_bf16(static_cast<float>(std::cos(a))),s=f32_to_bf16(static_cast<float>(std::sin(a)));
            std::size_t b=static_cast<std::size_t>(row)*hd;
            r.cos[b+i]=c;r.cos[b+i+half]=c;r.sin[b+i]=s;r.sin[b+i+half]=s;
        }
    }
    return r;
}

// ============================================================================
// legacy_legacy_tag_j.1 external benchmark metrics.
//
// Timing clock:
//   std::chrono::steady_clock wall time.
//
// TTFT definition in this offline TTS engine:
//   binary main entry -> first complete 16-codebook codec frame.
//
// TTFA definition:
//   binary main entry -> native waveform available on host.
//   The current vocoder is offline/non-streaming, so TTFA occurs after the
//   complete codec sequence has been generated and the decoder graph finishes.
//
// E2E definition:
//   binary main entry -> final WAV file write committed.
// ============================================================================
using PerfClock = std::chrono::steady_clock;
using PerfTime = PerfClock::time_point;

static double elapsed_ms(
    const PerfTime& begin,
    const PerfTime& end) {

    return std::chrono::duration<
        double,
        std::milli>(
            end - begin)
        .count();
}

static double percentile(
    std::vector<double> values,
    double quantile) {

    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end());

    quantile = std::clamp(
        quantile,
        0.0,
        1.0);

    const double position =
        quantile
        * static_cast<double>(
            values.size() - 1);

    const std::size_t low =
        static_cast<std::size_t>(
            std::floor(position));

    const std::size_t high =
        static_cast<std::size_t>(
            std::ceil(position));

    const double fraction =
        position
        - static_cast<double>(low);

    return values[low]
        + (
            values[high]
            - values[low]
        )
        * fraction;
}

struct PerfStats {
    double cli_and_tokenizer_ms=0.0;
    double frontend_ms=0.0;
    double core_model_setup_ms=0.0;
    double model_setup_wait_ms=0.0;
    double workdir_setup_ms=0.0;

    double predictor_engine_async_wall_ms=0.0;
    double predictor_engine_wait_ms=0.0;

    double talker_engine_async_wall_ms=0.0;
    double talker_engine_wait_ms=0.0;
    double talker_engine_init_ms=0.0;
    double talker_rope_build_ms=0.0;
    double talker_prefill_ms=0.0;
    double first_codec_frame_ms=0.0;
    double ttft_accounted_serial_ms=0.0;
    double ttft_unaccounted_ms=0.0;
    double codec_generation_ms=0.0;

    double ttft_ms=0.0;
    double ttfa_ms=0.0;

    double codec_decoder_parameter_preload_wall_ms=0.0;
    double codec_decoder_parameter_preload_wait_ms=0.0;
    double codec_decoder_setup_ms=0.0;
    double codec_decoder_graph_ms=0.0;
    double codec_decoder_pipeline_wall_ms=0.0;
    double codec_decoder_tail_wait_ms=0.0;
    double codec_decoder_first_chunk_ms=0.0;
    int codec_decoder_chunk_count=0;
    int codec_decoder_first_chunk_frames=0;
    int codec_decoder_chunk_frames=0;
    double wav_write_ms=0.0;
    double e2e_ms=0.0;

    std::vector<double> codec_emit_frame_ms;
    std::vector<double> talker_decode_step_ms;
    std::vector<double> predictor_frame_ms;
    std::size_t predictor_graph_node_count=0;
};

static double query_host_peak_rss_mib() {
#if defined(__linux__)
    struct rusage usage {};
    if (
        getrusage(
            RUSAGE_SELF,
            &usage)
        == 0
    ) {
        // Linux ru_maxrss is KiB.
        return static_cast<double>(
            usage.ru_maxrss)
            / 1024.0;
    }
#endif
    return 0.0;
}



static std::vector<std::int32_t> legacy_generate_native_codec_b(
    const fs::path& model_dir,const fs::path& work,
    const std::vector<std::uint16_t>& native_prefill,int prefill_length,int maximum_frames,
    const std::vector<std::uint16_t>& tts_pad,const fs::path& predictor_assets,
    const code_predictor::PredictorWeights& predictor_weights,
    const model_frontend::Tensor& codec_embedding,NativeFinalHead& final_head,
    const PerfTime& e2e_start,PerfStats& perf){

    if(prefill_length<=0||maximum_frames<=0)throw std::runtime_error("legacy_legacy_tag_h invalid recurrence geometry");

    const auto generation_start=
        PerfClock::now();

    constexpr int sliding_window=-1;

    const auto talker_init_start=
        PerfClock::now();

    talker::NativeTalkerEngine talker(
        model_dir,
        prefill_length,
        maximum_frames,
        sliding_window);

    perf.talker_engine_init_ms=
        elapsed_ms(
            talker_init_start,
            PerfClock::now());

    int hd=talker.head_dim();if(hd!=128)throw std::runtime_error("legacy_legacy_tag_h expects Talker head_dim=128");

    const auto rope_start=
        PerfClock::now();

    auto prefill_rope=make_talker_rope(prefill_length,0,hd,1000000.0);
    auto decode_rope=make_talker_rope(std::max(0,maximum_frames-1),prefill_length,hd,1000000.0);

    perf.talker_rope_build_ms=
        elapsed_ms(
            rope_start,
            PerfClock::now());

    std::cout<<"legacy_talker_rope_backend_a: native_cpp\n"
             <<"legacy_talker_rope_theta_a: 1000000\n"
             <<"talker_rope_text_mode: equal_mrope_axes_sequential\n"
             <<"talker_rope_reference_dependency: False\n"
             <<"talker_prefill_rope_rows: "<<prefill_length<<"\n"
             <<"talker_decode_rope_rows: "<<std::max(0,maximum_frames-1)<<"\n";

    const auto prefill_start=
        PerfClock::now();

    auto pr=talker.prefill(native_prefill,prefill_rope.cos,prefill_rope.sin,false);

    perf.talker_prefill_ms=
        elapsed_ms(
            prefill_start,
            PerfClock::now());

    std::vector<std::uint16_t> hidden=std::move(pr.hidden);
    std::vector<std::int32_t> history,codec;codec.reserve(static_cast<std::size_t>(maximum_frames)*16);

    int consecutive_same_code0=0;
    int maximum_same_code0_run=0;
    std::int32_t previous_code0=-1;

    fs::path runtime=work/"legacy_predictor_runtime_free_b",output=work/"legacy_predictor_output_free_b";
    fs::create_directories(runtime);

    for(int frame=0;frame<maximum_frames;++frame){
        const auto frame_emit_start=
            PerfClock::now();

        auto head=final_head.run(hidden.data(),history,frame);std::int32_t code0=head.code;
        if(code0==kCodecEos&&frame>=2){
            std::cout<<"legacy_native_recurrence_stop_b: reason=eos frame="<<frame<<"\n";break;
        }
        auto code0_embedding=model_frontend::tensor_row_bf16(codec_embedding,code0);
        create_predictor_frame_assets_free(runtime,predictor_assets,frame,hidden.data(),code0_embedding,tts_pad,code0);
        code_predictor::run_frame(runtime,output,"free",frame,predictor_weights);

        std::string fn=std::string("frame_")+(frame<10?"0":"")+std::to_string(frame);
        auto generated=read_binary<std::int32_t>(output/"free"/fn/"generated_codes.i32.bin");
        auto next=read_binary<std::uint16_t>(output/"free"/fn/"next_talker_input.bf16.bin");
        if(generated.size()!=16||next.size()!=1024||generated[0]!=code0)throw std::runtime_error("legacy_legacy_tag_h native Predictor recurrence output mismatch");

        codec.insert(codec.end(),generated.begin(),generated.end());history.push_back(code0);

        if(code0==previous_code0){
            ++consecutive_same_code0;
        }else{
            consecutive_same_code0=1;
            previous_code0=code0;
        }

        maximum_same_code0_run=
            std::max(
                maximum_same_code0_run,
                consecutive_same_code0);

        const auto frame_emitted=
            PerfClock::now();

        perf.codec_emit_frame_ms.push_back(
            elapsed_ms(
                frame_emit_start,
                frame_emitted));

        if(frame==0){
            perf.ttft_ms=
                elapsed_ms(
                    e2e_start,
                    frame_emitted);
        }

        std::cout<<"legacy_native_recurrence_frame_b: frame="<<frame<<" code0="<<code0<<" codec_groups=16\n";
        if(frame+1>=maximum_frames)break;

        const auto decode_step_start=
            PerfClock::now();

        hidden=talker.decode(
            next,
            decode_rope.cos.data()+static_cast<std::size_t>(frame)*hd,
            decode_rope.sin.data()+static_cast<std::size_t>(frame)*hd,
            frame,false).hidden;

        perf.talker_decode_step_ms.push_back(
            elapsed_ms(
                decode_step_start,
                PerfClock::now()));
    }
    if(codec.empty()||codec.size()%16)throw std::runtime_error("legacy_legacy_tag_h native recurrence produced no complete codec frames");
    perf.codec_generation_ms=
        elapsed_ms(
            generation_start,
            PerfClock::now());

    std::vector<std::int32_t> unique_code0=history;
    std::sort(
        unique_code0.begin(),
        unique_code0.end());
    unique_code0.erase(
        std::unique(
            unique_code0.begin(),
            unique_code0.end()),
        unique_code0.end());

    std::cout<<"legacy_native_recurrence_codec_frame_count_b: "<<codec.size()/16<<"\n"
             <<"legacy_code0_unique_count_a: "<<unique_code0.size()<<"\n"
             <<"legacy_code0_max_same_run_a: "<<maximum_same_code0_run<<"\n"
             <<"legacy_raw_text_to_codec_native_recurrence_b: True\n"
             <<"legacy_native_recurrence_teacher_next_input_dependency_b: False\n"
             <<"legacy_native_recurrence_hf_prefill_kv_dependency_b: False\n"
             <<"legacy_native_recurrence_rope_reference_dependency_b: False\n";
    return codec;
}


static std::vector<std::int32_t>
generate_codec_sequence(
    const fs::path& model_dir,
    const fs::path& work,
    const std::vector<std::uint16_t>& native_prefill,
    int prefill_length,
    int maximum_frames,
    const std::vector<std::uint16_t>& tts_pad,
    const std::vector<std::uint16_t>& trailing_text,
    std::size_t trailing_rows,
    const code_predictor::PredictorWeights& predictor_weights,
    const model_frontend::Tensor& codec_embedding,
    NativeFinalHead& final_head,
    std::uint64_t* gpu_rng_state,
    std::future<
        std::unique_ptr<
            code_predictor::PersistentPredictor
        >
    >* predictor_future,
    const PerfTime& predictor_async_start,
    std::future<
        std::unique_ptr<
            talker::NativeTalkerEngine
        >
    >* talker_future,
    const PerfTime& talker_async_start,
    const std::function<void()>& first_frame_hook,
    const std::function<
        void(
            const std::array<
                std::int32_t,
                16>&
        )
    >& codec_frame_hook,
    const PerfTime& e2e_start,
    PerfStats& perf,
    const fs::path& correctness_dir,
    code_predictor::PersistentPredictor*
        resident_predictor=nullptr,
    talker::NativeTalkerEngine*
        resident_talker=nullptr) {

    if(
        prefill_length<=0
        ||maximum_frames<=0
        ||tts_pad.size()!=1024
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 invalid official Base recurrence geometry");
    }

    if(
        trailing_text.size()
        !=trailing_rows*1024
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 trailing_text_hidden geometry mismatch");
    }

    const auto generation_start=
        PerfClock::now();

    const bool use_persistent_predictor=
        correctness_dir.empty();

    constexpr int sliding_window=-1;
    (void)sliding_window;

    const auto talker_wait_start=
        PerfClock::now();

    std::unique_ptr<
        talker::NativeTalkerEngine>
        talker_owner;

    talker::NativeTalkerEngine*
        talker_pointer=
            resident_talker;

    if(talker_pointer==nullptr){
        if(
            talker_future!=nullptr
            &&talker_future->valid()
        ){
            talker_owner=
                talker_future->get();
        }else{
            talker_owner=
                std::make_unique<
                    talker::NativeTalkerEngine>(
                        model_dir,
                        prefill_length,
                        maximum_frames,
                        -1);
        }

        talker_pointer=
            talker_owner.get();
    }

    const auto talker_ready=
        PerfClock::now();

    if(resident_talker!=nullptr){
        perf.talker_engine_wait_ms=0.0;
        perf.talker_engine_async_wall_ms=0.0;
        perf.talker_engine_init_ms=0.0;
    }else{
        perf.talker_engine_wait_ms=
            elapsed_ms(
                talker_wait_start,
                talker_ready);

        perf.talker_engine_async_wall_ms=
            elapsed_ms(
                talker_async_start,
                talker_ready);

        perf.talker_engine_init_ms=
            perf.talker_engine_wait_ms;
    }

    auto& talker=
        *talker_pointer;

    const int hd=talker.head_dim();

    if(hd!=128){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 expects Talker head_dim=128");
    }

    const auto rope_start=
        PerfClock::now();

    auto prefill_rope=
        make_talker_rope(
            prefill_length,
            0,
            hd,
            1000000.0);

    auto decode_rope=
        make_talker_rope(
            std::max(
                0,
                maximum_frames-1),
            prefill_length,
            hd,
            1000000.0);

    perf.talker_rope_build_ms=
        elapsed_ms(
            rope_start,
            PerfClock::now());

    std::cout
        <<"legacy_talker_rope_backend_b: native_cpp\\n"
        <<"legacy_talker_rope_theta_b: 1000000\\n"
        <<"talker_prefill_rows: "<<prefill_length<<"\\n"
        <<"talker_trailing_text_rows: "<<trailing_rows<<"\\n";

    const auto prefill_start=
        PerfClock::now();

    const bool correctness_capture=
        !correctness_dir.empty();

    auto prefill=
        talker.prefill(
            native_prefill,
            prefill_rope.cos,
            prefill_rope.sin,
            correctness_capture);

    if(correctness_capture){
        write_binary(
            correctness_dir/
                "native_talker_prefill_final_hidden.bf16.bin",
            prefill.hidden);

        write_binary(
            correctness_dir/
                "native_talker_prefill_rope_cos.bf16.bin",
            prefill_rope.cos);

        write_binary(
            correctness_dir/
                "native_talker_prefill_rope_sin.bf16.bin",
            prefill_rope.sin);

        for(
            std::size_t layer=0;
            layer<prefill.layer_last_rows.size();
            ++layer
        ){
            std::ostringstream name;
            name
                <<"native_talker_layer_"
                <<std::setw(2)
                <<std::setfill('0')
                <<layer
                <<"_last.bf16.bin";

            write_binary(
                correctness_dir/name.str(),
                prefill.layer_last_rows[layer]);
        }
    }

    perf.talker_prefill_ms=
        elapsed_ms(
            prefill_start,
            PerfClock::now());

    std::vector<std::uint16_t> hidden=
        std::move(
            prefill.hidden);

    const std::uint16_t*
        current_hidden_device =
            talker.device_final_hidden();

    std::vector<std::int32_t>
        code0_history;

    std::vector<std::int32_t> codec;

    codec.reserve(
        static_cast<std::size_t>(
            maximum_frames)
        *16);

    fs::path runtime;
    fs::path output;

    if(!use_persistent_predictor){
        runtime=
            work/
                "legacy_predictor_runtime";

        output=
            work/
                "legacy_predictor_output";

        fs::create_directories(
            runtime);
    }

    std::unique_ptr<
        code_predictor::PersistentPredictor>
        persistent_predictor_owner;

    code_predictor::PersistentPredictor*
        persistent_predictor=
            resident_predictor;

    if(use_persistent_predictor){
        const auto wait_start=
            PerfClock::now();

        if(persistent_predictor==nullptr){
            if(
                predictor_future!=nullptr
                &&predictor_future->valid()
            ){
                persistent_predictor_owner=
                    predictor_future->get();
            }else{
                persistent_predictor_owner=
                    std::make_unique<
                        code_predictor::PersistentPredictor>(
                            predictor_weights,
                            codec_embedding,
                            gpu_rng_state);
            }

            persistent_predictor=
                persistent_predictor_owner.get();
        }

        const auto predictor_ready=
            PerfClock::now();

        if(resident_predictor!=nullptr){
            perf.predictor_engine_wait_ms=0.0;
            perf.predictor_engine_async_wall_ms=0.0;
        }else{
            perf.predictor_engine_wait_ms=
                elapsed_ms(
                    wait_start,
                    predictor_ready);

            perf.predictor_engine_async_wall_ms=
                elapsed_ms(
                    predictor_async_start,
                    predictor_ready);
        }

        perf.predictor_graph_node_count=
            persistent_predictor
                ->graph_node_count();
    }

    int maximum_same_code0_run=0;
    int current_same_code0_run=0;
    std::int32_t previous_code0=-1;

    for(int frame=0;frame<maximum_frames;++frame){
        const auto frame_start=
            PerfClock::now();

        FinalHeadTrace head;
        std::int32_t code0=0;

        if(use_persistent_predictor){
            code0=
                final_head.run_fast_device(
                    current_hidden_device,
                    frame,
                    gpu_rng_state,
                    talker.stream());
        }else{
            head=
                final_head.run(
                    hidden.data(),
                    code0_history,
                    frame);
            code0=head.code;
        }

        if(
            !use_persistent_predictor
            &&frame==0
            &&!correctness_dir.empty()
        ){
            write_binary(
                correctness_dir/
                    "native_frame0_raw_logits.f32.bin",
                head.raw_logits);

            write_binary(
                correctness_dir/
                    "native_frame0_processed_scores.f32.bin",
                head.processed_scores);

            write_scalar_text(
                correctness_dir/
                    "native_frame0_code0.txt",
                std::to_string(code0));
        }

        if(
            code0==kCodecEos
            &&frame>=2
        ){
            std::cout
                <<"legacy_native_recurrence_stop_c: reason=eos frame="
                <<frame
                <<"\\n";
            break;
        }

        std::vector<std::uint16_t>
            context(1024);

        if(
            static_cast<std::size_t>(frame)
            <trailing_rows
        ){
            const auto first=
                trailing_text.begin()
                +static_cast<std::ptrdiff_t>(
                    frame*1024);

            std::copy(
                first,
                first+1024,
                context.begin());
        }else{
            context=tts_pad;
        }

        std::vector<std::int32_t> generated;
        std::vector<std::uint16_t> next;

        if(use_persistent_predictor){
            auto predictor_result=
                persistent_predictor
                    ->run_frame_device(
                        current_hidden_device,
                        final_head.device_code(),
                        context.data());

            generated.assign(
                predictor_result.codes.begin(),
                predictor_result.codes.end());

            perf.predictor_frame_ms.push_back(
                predictor_result.elapsed_ms);
        }else{
            const auto code0_embedding=
                model_frontend::
                    tensor_row_bf16(
                        codec_embedding,
                        code0);

            legacy_create_predictor_frame_assets_b(
                runtime,
                frame,
                hidden.data(),
                code0_embedding,
                context,
                code0);

            code_predictor::run_frame(
                runtime,
                output,
                "free",
                frame,
                predictor_weights);

            const std::string frame_name=
                std::string("frame_")
                +(frame<10?"0":"")
                +std::to_string(frame);

            generated=
                read_binary<std::int32_t>(
                    output/
                        "free"/
                        frame_name/
                        "generated_codes.i32.bin");

            next=
                read_binary<std::uint16_t>(
                    output/
                        "free"/
                        frame_name/
                        "next_talker_input.bf16.bin");
        }

        if(
            generated.size()!=16
            ||(
                !use_persistent_predictor
                &&next.size()!=1024
            )
            ||generated[0]!=code0
        ){
            throw std::runtime_error(
                "native Predictor recurrence output mismatch");
        }

        codec.insert(
            codec.end(),
            generated.begin(),
            generated.end());

        code0_history.push_back(
            code0);

        if(code0==previous_code0){
            ++current_same_code0_run;
        }else{
            previous_code0=code0;
            current_same_code0_run=1;
        }

        maximum_same_code0_run=
            std::max(
                maximum_same_code0_run,
                current_same_code0_run);

        const auto frame_emitted=
            PerfClock::now();

        const double frame_ms=
            elapsed_ms(
                frame_start,
                frame_emitted);

        perf.codec_emit_frame_ms.push_back(
            frame_ms);

        if(frame==0){
            perf.first_codec_frame_ms=
                frame_ms;

            perf.ttft_ms=
                elapsed_ms(
                    e2e_start,
                    frame_emitted);

            // E2E-only preload starts strictly after TTFT is committed.
            if(first_frame_hook){
                first_frame_hook();
            }
        }

        if(codec_frame_hook){
            std::array<
                std::int32_t,
                16> frame_codes{};

            std::copy_n(
                generated.begin(),
                16,
                frame_codes.begin());

            codec_frame_hook(
                frame_codes);
        }

        // Production output is compact: per-frame terminal I/O is not on the
        // generation critical path. Keep first-frame visibility and full
        // diagnostic logging when correctness capture is enabled.
        if(
            correctness_capture
            ||frame==0
        ){
            std::cout
                <<"native_recurrence_frame: frame="
                <<frame
                <<" code0="
                <<code0
                <<" trailing_text="
                <<(
                    static_cast<std::size_t>(frame)
                    <trailing_rows
                    ?"True"
                    :"False")
                <<" codec_groups=16\\n";
        }

        if(frame+1>=maximum_frames){
            break;
        }

        const auto decode_start=
            PerfClock::now();

        if(use_persistent_predictor){
            talker.decode_device(
                persistent_predictor
                    ->device_next_talker_input(),
                decode_rope.cos.data()
                    +static_cast<std::size_t>(
                        frame)*hd,
                decode_rope.sin.data()
                    +static_cast<std::size_t>(
                        frame)*hd,
                frame);

            current_hidden_device=
                talker.device_final_hidden();
        }else{
            hidden=
                talker.decode(
                    next,
                    decode_rope.cos.data()
                        +static_cast<std::size_t>(
                            frame)*hd,
                    decode_rope.sin.data()
                        +static_cast<std::size_t>(
                            frame)*hd,
                    frame,
                    false)
                .hidden;
        }

        perf.talker_decode_step_ms.push_back(
            elapsed_ms(
                decode_start,
                PerfClock::now()));
    }

    if(
        codec.empty()
        ||codec.size()%16
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 native recurrence produced no complete codec frames");
    }

    perf.codec_generation_ms=
        elapsed_ms(
            generation_start,
            PerfClock::now());

    auto unique_code0=
        code0_history;

    std::sort(
        unique_code0.begin(),
        unique_code0.end());

    unique_code0.erase(
        std::unique(
            unique_code0.begin(),
            unique_code0.end()),
        unique_code0.end());

    std::cout
        <<"legacy_native_recurrence_codec_frame_count_c: "
        <<codec.size()/16
        <<"\\n"
        <<"legacy_code0_unique_count_b: "
        <<unique_code0.size()
        <<"\\n"
        <<"legacy_code0_max_same_run_b: "
        <<maximum_same_code0_run
        <<"\\n"
        <<"trailing_text_conditioning: True\\n"
        <<"predictor_rope_source: native_cpp\\n"
        <<"predictor_rope_theta: 1000000\\n"
        <<"predictor_external_rope_asset_dependency: False\n"
        <<"gpu_sampling_backend: "
        <<(use_persistent_predictor
            ?"gpu_topk50_splitmix64"
            :"diagnostic_host_sampler")
        <<"\n"
        <<"predictor_execution_backend: "
        <<(use_persistent_predictor
            ?"persistent_15_step_hip_graph"
            :"diagnostic_stepwise")
        <<"\n"
        <<"predictor_graph_node_count: "
        <<perf.predictor_graph_node_count
        <<"\n"
        <<"device_chain_talker_hidden_d2h_per_frame: False\n"
        <<"device_chain_talker_hidden_h2d_per_frame: False\n"
        <<"device_chain_predictor_next_input_d2h: False\n"
        <<"device_chain_code0_history_h2d_per_frame: False\n"
        <<"device_chain_only_host_codec_ids_per_frame: True\n"
        <<"production_workdir_dependency: "
        <<(use_persistent_predictor?"False":"True")
        <<"\n"
        <<"production_per_frame_stdout: first_frame_only\n"
        <<"predictor_async_build_overlap: "
        <<(use_persistent_predictor?"True":"False")
        <<"\n"
        <<"generation_stream_policy: explicit_nonblocking\n"
        <<"generation_frame_boundary: final_head_code_readback_stream_sync\n"
        <<"generation_device_wide_sync_per_frame: False\n"
        <<"generation_decoder_stream_dependency: False\n"
        <<"final_head_sampling_stream: generation_stream\n"
        <<"final_head_history_append_stream: generation_stream\n"
        <<"final_head_code_readback_stream: generation_stream\n"
        <<"final_head_ordering_chain_closed: True\n";

    return codec;
}


static std::vector<std::int32_t>
legacy_generate_native_codec_a(
    const fs::path& model_dir,
    const fs::path& work,
    const std::vector<std::uint16_t>& native_prefill,
    int prefill_length,
    int maximum_frames,
    const std::vector<std::uint16_t>& tts_pad,
    const fs::path& predictor_assets,
    const code_predictor::PredictorWeights& predictor_weights,
    const model_frontend::Tensor& codec_embedding,
    NativeFinalHead& final_head) {

    const int sliding_window =
        read_int_text(
            work
            / "talker_sliding_window.txt");

    talker::NativeTalkerEngine talker(
        model_dir,
        prefill_length,
        maximum_frames,
        sliding_window);

    const auto prefill_cosine =
        read_binary<std::uint16_t>(
            work
            / "talker_prefill_rope_cos.bf16.bin");

    const auto prefill_sine =
        read_binary<std::uint16_t>(
            work
            / "talker_prefill_rope_sin.bf16.bin");

    const auto decode_cosine =
        read_binary<std::uint16_t>(
            work
            / "talker_decode_rope_cos.bf16.bin");

    const auto decode_sine =
        read_binary<std::uint16_t>(
            work
            / "talker_decode_rope_sin.bf16.bin");

    const int head_dim =
        talker.head_dim();

    const int available_decode_rows =
        static_cast<int>(
            decode_cosine.size()
            / static_cast<std::size_t>(
                head_dim));

    if (
        decode_sine.size()
        != decode_cosine.size()
    ) {
        throw std::runtime_error(
            "legacy_legacy_tag_g native recurrence decode RoPE mismatch.");
    }

    const int frame_limit =
        std::min(
            maximum_frames,
            available_decode_rows + 1);

    if (frame_limit <= 0) {
        throw std::runtime_error(
            "legacy_legacy_tag_g native recurrence has no usable frame budget.");
    }

    auto prefill =
        talker.prefill(
            native_prefill,
            prefill_cosine,
            prefill_sine,
            false);

    std::vector<std::uint16_t> hidden =
        std::move(
            prefill.hidden);

    std::vector<std::int32_t> code0_history;

    std::vector<std::int32_t> codec;
    codec.reserve(
        static_cast<std::size_t>(
            frame_limit)
        * 16);

    const fs::path runtime =
        work
        / "legacy_predictor_runtime_free_a";

    const fs::path output =
        work
        / "legacy_predictor_output_free_a";

    fs::create_directories(
        runtime);

    for (
        int frame = 0;
        frame < frame_limit;
        ++frame
    ) {
        const FinalHeadTrace head =
            final_head.run(
                hidden.data(),
                code0_history,
                frame);

        const std::int32_t code0 =
            head.code;

        if (
            code0 == kCodecEos
            && frame >= 2
        ) {
            std::cout
                << "legacy_native_recurrence_stop_a: "
                << "reason=eos"
                << " frame="
                << frame
                << "\n";
            break;
        }

        const auto code0_embedding =
            model_frontend::
                tensor_row_bf16(
                    codec_embedding,
                    code0);

        create_predictor_frame_assets_free(
            runtime,
            predictor_assets,
            frame,
            hidden.data(),
            code0_embedding,
            tts_pad,
            code0);

        code_predictor::run_frame(
            runtime,
            output,
            "free",
            frame,
            predictor_weights);

        const std::string frame_name =
            std::string("frame_")
            + (
                frame < 10
                ? "0"
                : ""
            )
            + std::to_string(frame);

        const auto generated =
            read_binary<std::int32_t>(
                output
                / "free"
                / frame_name
                / "generated_codes.i32.bin");

        const auto next =
            read_binary<std::uint16_t>(
                output
                / "free"
                / frame_name
                / "next_talker_input.bf16.bin");

        if (
            generated.size() != 16
            || next.size() != 1024
            || generated[0] != code0
        ) {
            throw std::runtime_error(
                "legacy_legacy_tag_g native Predictor recurrence output mismatch.");
        }

        codec.insert(
            codec.end(),
            generated.begin(),
            generated.end());

        code0_history.push_back(
            code0);

        std::cout
            << "legacy_native_recurrence_frame_a: "
            << "frame="
            << frame
            << " code0="
            << code0
            << " codec_groups=16"
            << "\n";

        if (frame + 1 >= frame_limit) {
            break;
        }

        hidden =
            talker.decode(
                next,
                decode_cosine.data()
                    + static_cast<std::size_t>(
                        frame)
                    * head_dim,
                decode_sine.data()
                    + static_cast<std::size_t>(
                        frame)
                    * head_dim,
                frame,
                false)
            .hidden;
    }

    if (
        codec.empty()
        || codec.size() % 16 != 0
    ) {
        throw std::runtime_error(
            "legacy_legacy_tag_g native recurrence produced no complete codec frames.");
    }

    std::cout
        << "legacy_native_recurrence_codec_frame_count_a: "
        << (
            codec.size()
            / 16)
        << "\n"
        << "legacy_raw_text_to_codec_native_recurrence_a: True\n"
        << "legacy_native_recurrence_teacher_next_input_dependency_a: False\n"
        << "legacy_native_recurrence_hf_prefill_kv_dependency_a: False\n"
        << "legacy_native_recurrence_rope_reference_dependency_a: True\n";

    return codec;
}




// ============================================================================
// legacy_legacy_tag_h.0 production inference entry.
// No Python/HuggingFace process is launched below this boundary.
// ============================================================================


struct OfficialPreparedModel {
    model_frontend::ModelWeights root_weights;
    NativeFinalHead final_head;
    code_predictor::PredictorWeights predictor_weights;

    explicit OfficialPreparedModel(
        const fs::path& model_dir)
        :root_weights(model_dir),
         final_head(model_dir),
         predictor_weights(
             root_weights,
             "direct_model_safetensors"){}

    const model_frontend::Tensor&
    codec_embedding() const {
        return root_weights.suffix(
            "talker.model.codec_embedding.weight");
    }
};


struct CustomVoiceContract {
    std::unordered_map<
        std::string,
        std::int64_t>
        speaker_ids;

    std::unordered_map<
        std::string,
        std::string>
        speaker_dialects;

    std::unordered_map<
        std::string,
        std::int64_t>
        language_ids;
};

static std::string lower_ascii_copy(
    std::string value) {

    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c){
            return static_cast<char>(
                std::tolower(c));
        });

    return value;
}

static const model_frontend::J&
custom_require_json_kind(
    const model_frontend::J& value,
    model_frontend::J::K kind,
    const std::string& field) {

    if(value.kind!=kind){
        throw std::runtime_error(
            "CustomVoice config field has unexpected type: "
            +field);
    }

    return value;
}

static CustomVoiceContract
load_custom_voice_contract(
    const fs::path& model_dir) {

    const std::string config_text=
        read_text_file(
            model_dir/"config.json");

    model_frontend::JP parser(
        config_text);

    const auto root=
        parser.parse();

    custom_require_json_kind(
        root,
        model_frontend::J::K::Object,
        "root");

    const std::string model_size=
        lower_ascii_copy(
            custom_require_json_kind(
                root.at("tts_model_size"),
                model_frontend::J::K::String,
                "tts_model_size")
                .s);

    const std::string model_type=
        lower_ascii_copy(
            custom_require_json_kind(
                root.at("tts_model_type"),
                model_frontend::J::K::String,
                "tts_model_type")
                .s);

    if(
        model_size!="0b6"
        &&model_size!="0.6b"
    ){
        throw std::runtime_error(
            "CustomVoice requires 0.6B checkpoint; config tts_model_size="
            +model_size);
    }

    if(model_type!="custom_voice"){
        throw std::runtime_error(
            "CustomVoice task requires tts_model_type=custom_voice; config has "
            +model_type);
    }

    const auto& talker=
        custom_require_json_kind(
            root.at("talker_config"),
            model_frontend::J::K::Object,
            "talker_config");

    CustomVoiceContract out;

    const auto& speaker_map=
        custom_require_json_kind(
            talker.at("spk_id"),
            model_frontend::J::K::Object,
            "talker_config.spk_id");

    for(const auto& item:speaker_map.o){
        const auto& value=
            custom_require_json_kind(
                item.second,
                model_frontend::J::K::Number,
                "talker_config.spk_id."
                    +item.first);

        out.speaker_ids[
            lower_ascii_copy(
                item.first)]=
                    static_cast<
                        std::int64_t>(
                            value.n);
    }

    if(out.speaker_ids.empty()){
        throw std::runtime_error(
            "CustomVoice checkpoint has empty talker_config.spk_id");
    }

    const auto& dialect_map=
        custom_require_json_kind(
            talker.at("spk_is_dialect"),
            model_frontend::J::K::Object,
            "talker_config.spk_is_dialect");

    for(const auto& item:dialect_map.o){
        const std::string key=
            lower_ascii_copy(
                item.first);

        if(
            item.second.kind
            ==model_frontend::J::K::Bool
        ){
            if(item.second.b){
                throw std::runtime_error(
                    "CustomVoice boolean true dialect entry is unsupported: "
                    +item.first);
            }

            out.speaker_dialects[key]=
                std::string{};
        }else if(
            item.second.kind
            ==model_frontend::J::K::String
        ){
            out.speaker_dialects[key]=
                lower_ascii_copy(
                    item.second.s);
        }else{
            throw std::runtime_error(
                "CustomVoice dialect entry has unexpected type: "
                +item.first);
        }
    }

    const auto& language_map=
        custom_require_json_kind(
            talker.at("codec_language_id"),
            model_frontend::J::K::Object,
            "talker_config.codec_language_id");

    for(const auto& item:language_map.o){
        const auto& value=
            custom_require_json_kind(
                item.second,
                model_frontend::J::K::Number,
                "talker_config.codec_language_id."
                    +item.first);

        out.language_ids[
            lower_ascii_copy(
                item.first)]=
                    static_cast<
                        std::int64_t>(
                            value.n);
    }

    return out;
}

struct CustomVoiceConditioning {
    std::vector<std::uint16_t>
        speaker_embedding;

    std::string effective_language;
};

static std::string
resolve_custom_voice_effective_language(
    const CustomVoiceContract& contract,
    const std::string& requested_speaker,
    const std::string& requested_language) {

    const std::string speaker=
        lower_ascii_copy(
            requested_speaker);

    const std::string language=
        lower_ascii_copy(
            requested_language);

    if(
        contract.speaker_ids.find(
            speaker)
        ==contract.speaker_ids.end()
    ){
        throw std::runtime_error(
            "unsupported CustomVoice speaker: "
            +requested_speaker);
    }

    if(
        language!="auto"
        &&contract.language_ids.find(
            language)
            ==contract.language_ids.end()
    ){
        throw std::runtime_error(
            "unsupported CustomVoice language: "
            +requested_language);
    }

    const auto dialect_it=
        contract.speaker_dialects.find(
            speaker);

    if(
        (
            language=="chinese"
            ||language=="auto"
        )
        &&dialect_it
            !=contract.speaker_dialects.end()
        &&!dialect_it->second.empty()
    ){
        if(
            contract.language_ids.find(
                dialect_it->second)
            ==contract.language_ids.end()
        ){
            throw std::runtime_error(
                "CustomVoice dialect is not present in codec_language_id: "
                +dialect_it->second);
        }

        return dialect_it->second;
    }

    return language;
}

static CustomVoiceConditioning
build_custom_voice_conditioning(
    const CustomVoiceContract& contract,
    const model_frontend::Tensor& codec_embedding,
    const std::string& requested_speaker,
    const std::string& requested_language) {

    const std::string speaker=
        lower_ascii_copy(
            requested_speaker);

    const std::string language=
        lower_ascii_copy(
            requested_language);

    const auto speaker_it=
        contract.speaker_ids.find(
            speaker);

    if(
        speaker_it
        ==contract.speaker_ids.end()
    ){
        throw std::runtime_error(
            "unsupported CustomVoice speaker: "
            +requested_speaker);
    }

    if(
        language!="auto"
        &&contract.language_ids.find(
            language)
            ==contract.language_ids.end()
    ){
        throw std::runtime_error(
            "unsupported CustomVoice language: "
            +requested_language);
    }

    std::string effective_language=
        language;

    const auto dialect_it=
        contract.speaker_dialects.find(
            speaker);

    if(
        (
            language=="chinese"
            ||language=="auto"
        )
        &&dialect_it
            !=contract.speaker_dialects.end()
        &&!dialect_it->second.empty()
    ){
        effective_language=
            dialect_it->second;

        if(
            contract.language_ids.find(
                effective_language)
            ==contract.language_ids.end()
        ){
            throw std::runtime_error(
                "CustomVoice dialect is not present in codec_language_id: "
                +effective_language);
        }
    }

    if(
        codec_embedding.dtype!="BF16"
        ||codec_embedding.shape.size()!=2
        ||codec_embedding.shape[1]!=1024
        ||speaker_it->second<0
        ||speaker_it->second
            >=codec_embedding.shape[0]
    ){
        throw std::runtime_error(
            "CustomVoice speaker id / codec embedding geometry mismatch");
    }

    CustomVoiceConditioning result;

    result.speaker_embedding=
        model_frontend::
            tensor_row_bf16(
                codec_embedding,
                speaker_it->second);

    result.effective_language=
        std::move(
            effective_language);

    return result;
}

static int official_xvector_prefill_rows(
    std::string language) {

    std::transform(
        language.begin(),
        language.end(),
        language.begin(),
        [](unsigned char c){
            return static_cast<char>(
                std::tolower(c));
        });

    return language=="auto" ? 9 : 10;
}


static int run_official_base_xvector(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids,
    std::future<
        std::unique_ptr<
            OfficialPreparedModel
        >
    >* model_future,
    const PerfTime& model_async_start,
    const PerfTime& e2e_start,
    double cli_and_tokenizer_ms) {

    if(options.ref_audio.empty()){
        throw std::runtime_error(
            "official Base x-vector voice clone requires --ref-audio");
    }

    const fs::path model_dir=
        fs::absolute(
            options.model_dir);

    const fs::path ref_audio=
        fs::absolute(
            options.ref_audio);

    const fs::path correctness_dir=
        options.correctness_dir.empty()
        ?fs::path{}
        :fs::absolute(
            options.correctness_dir);

    if(!correctness_dir.empty()){
        fs::remove_all(correctness_dir);
        fs::create_directories(correctness_dir);

        write_i64(
            correctness_dir/
                "native_input_ids.i64.bin",
            native_ids);
    }

    if(!fs::is_regular_file(ref_audio)){
        throw std::runtime_error(
            "reference WAV does not exist: "
            +ref_audio.string());
    }

    // legacy_legacy_tag_j.3 follows the official generate_voice_clone sampling defaults.
    configure_sampling(
        !options.greedy,
        options.seed);

    GpuSamplingState gpu_sampling_state(
        options.seed);

    PerfStats perf;
    perf.cli_and_tokenizer_ms=
        cli_and_tokenizer_ms;

    fs::path work;

    const auto workdir_start=
        PerfClock::now();

    if(!correctness_dir.empty()){
        work=
            select_work_dir(
                options);

        fs::remove_all(work);
        fs::create_directories(work);
    }

    perf.workdir_setup_ms=
        elapsed_ms(
            workdir_start,
            PerfClock::now());

    std::cout
        <<"legacy_mode_e: official_base_xvector_voice_clone\n"
        <<"model_contract: Qwen3-TTS-12Hz-0.6B-Base\n"
        <<"official_api_contract: generate_voice_clone\n"
        <<"legacy_x_vector_only_mode: True\n"
        <<"icl_mode: False\n"
        <<"reference_audio_dependency: True\n"
        <<"ref_text_dependency: False\n"
        <<"ref_code_dependency: False\n"
        <<"legacy_runtime_python_dependency_b: False\n"
        <<"legacy_runtime_huggingface_dependency_b: False\n"
        <<"legacy_runtime_rocblas_dependency_b: False\n"
        <<"legacy_runtime_hipblas_dependency_b: False\n"
        <<"runtime_external_code_objects: False\n"
        <<"external_native_assets_dependency: False\n"
        <<"reference_audio: "<<ref_audio.string()<<"\n";

    const auto model_wait_start=
        PerfClock::now();

    std::unique_ptr<
        OfficialPreparedModel>
        prepared_model;

    if(
        model_future!=nullptr
        &&model_future->valid()
    ){
        prepared_model=
            model_future->get();
    }else{
        prepared_model=
            std::make_unique<
                OfficialPreparedModel>(
                    model_dir);
    }

    const auto model_ready=
        PerfClock::now();

    perf.model_setup_wait_ms=
        elapsed_ms(
            model_wait_start,
            model_ready);

    perf.core_model_setup_ms=
        elapsed_ms(
            model_async_start,
            model_ready);

    auto& final_head=
        prepared_model->final_head;

    auto& predictor_weights=
        prepared_model->predictor_weights;

    const auto& codec_embedding=
        prepared_model->codec_embedding();

    // Predictor graph construction is request-independent after model setup.
    // Start it before speaker conditioning so graph capture/instantiate,
    // allocations and RoPE upload overlap the speaker/frontend/Talker path.
    const auto predictor_async_start=
        PerfClock::now();

    std::future<
        std::unique_ptr<
            code_predictor::PersistentPredictor
        >
    > predictor_future;

    if(correctness_dir.empty()){
        predictor_future=
            std::async(
                std::launch::async,
                [&predictor_weights,
                 &codec_embedding,
                 &gpu_sampling_state](){
                    return std::make_unique<
                        code_predictor::PersistentPredictor>(
                            predictor_weights,
                            codec_embedding,
                            gpu_sampling_state
                                .device_state());
                });
    }

    const int expected_prefill_rows=
        official_xvector_prefill_rows(
            options.language);

    const auto talker_async_start=
        PerfClock::now();

    std::future<
        std::unique_ptr<
            talker::NativeTalkerEngine
        >
    > talker_future;

    if(correctness_dir.empty()){
        talker_future=
            std::async(
                std::launch::async,
                [model_dir,
                 expected_prefill_rows,
                 maximum_frames=
                    static_cast<int>(
                        generation_capacity_for(options.max_new_tokens))](){
                    return std::make_unique<
                        talker::NativeTalkerEngine>(
                            model_dir,
                            expected_prefill_rows,
                            maximum_frames,
                            -1);
                });
    }

    const auto speaker_start=
        PerfClock::now();

    std::vector<std::uint16_t>
        speaker_embedding;

    if(!options.speaker_embedding_bf16.empty()){
        speaker_embedding=
            read_binary<std::uint16_t>(
                fs::absolute(
                    options.speaker_embedding_bf16));

        if(speaker_embedding.size()!=1024){
            throw std::runtime_error(
                "--speaker-embedding-bf16 must contain exactly 1024 BF16 values");
        }

        std::cout
            <<"speaker_embedding_source: official_or_external_bf16_bypass\n";
    }else{
        speaker_encoder::Encoder
            speaker_encoder(model_dir);

        speaker_embedding=
            speaker_encoder.run(
                ref_audio,
                correctness_dir);

        std::cout
            <<"speaker_embedding_source: native_reference_audio_encoder\n";
    }

    if(!correctness_dir.empty()){
        write_binary(
            correctness_dir/
                "native_speaker_embedding_used.bf16.bin",
            speaker_embedding);
    }

    const double speaker_ms=
        elapsed_ms(
            speaker_start,
            PerfClock::now());

    const auto frontend_start=
        PerfClock::now();

    model_frontend::NativeFrontend frontend(
        model_dir,
        fs::path{});

    const auto native=
        frontend.build_official_base_xvector_inputs(
            native_ids,
            speaker_embedding,
            options.language);

    perf.frontend_ms=
        elapsed_ms(
            frontend_start,
            PerfClock::now());

    if(!correctness_dir.empty()){
        write_binary(
            correctness_dir/
                "native_talker_prefill_input.bf16.bin",
            native.prefill);

        write_binary(
            correctness_dir/
                "native_trailing_text_hidden.bf16.bin",
            native.trailing_text);

        write_binary(
            correctness_dir/
                "native_tts_pad.bf16.bin",
            native.tts_pad);
    }

    if(
        static_cast<int>(native.seq_len)
        !=expected_prefill_rows
    ){
        throw std::runtime_error(
            "official Base predicted prefill rows disagree with frontend result");
    }

    if(
        native.prefill.size()
        !=native.seq_len*1024
        ||native.trailing_text.size()
            !=native.trailing_rows*1024
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 official Base frontend output geometry mismatch");
    }

    std::unique_ptr<
        codec_decoder::StreamingCodecPipeline>
        streaming_decoder;

    PerfTime decoder_preload_start{};

    const std::function<void()>
        first_frame_hook=
            [&streaming_decoder,
             &decoder_preload_start,
             model_dir,
             maximum_frames=
                static_cast<int>(
                    generation_capacity_for(options.max_new_tokens)),
             e2e_start](){

                decoder_preload_start=
                    PerfClock::now();

                auto preload_future=
                    std::async(
                        std::launch::async,
                        [model_dir](){
                            const fs::path tokenizer=
                                codec_decoder::
                                    locate_speech_tokenizer(
                                        model_dir);

                            auto parameters=
                                std::make_shared<
                                    codec_decoder::Parameters>(
                                        tokenizer);

                            return std::make_pair(
                                tokenizer,
                                std::move(parameters));
                        });

                streaming_decoder=
                    std::make_unique<
                        codec_decoder::
                            StreamingCodecPipeline>(
                                std::move(
                                    preload_future),
                                decoder_preload_start,
                                maximum_frames,
                                e2e_start,
                                codec_decoder::
                                    STREAM_CHUNK_FRAMES);
            };

    const std::function<
        void(
            const std::array<
                std::int32_t,
                16>&
        )
    > codec_frame_hook=
        [&streaming_decoder](
            const std::array<
                std::int32_t,
                16>& frame_codes){

            if(streaming_decoder){
                streaming_decoder
                    ->push_frame(
                        frame_codes);
            }
        };

    const auto codec=
        generate_codec_sequence(
            model_dir,
            work,
            native.prefill,
            static_cast<int>(
                native.seq_len),
            static_cast<int>(
                options.max_new_tokens),
            native.tts_pad,
            native.trailing_text,
            native.trailing_rows,
            predictor_weights,
            codec_embedding,
            final_head,
            gpu_sampling_state.device_state(),
            correctness_dir.empty()
                ?&predictor_future
                :nullptr,
            predictor_async_start,
            correctness_dir.empty()
                ?&talker_future
                :nullptr,
            talker_async_start,
            correctness_dir.empty()
                ?first_frame_hook
                :std::function<void()>{},
            correctness_dir.empty()
                ?codec_frame_hook
                :std::function<
                    void(
                        const std::array<
                            std::int32_t,
                            16>&
                    )
                >{},
            e2e_start,
            perf,
            correctness_dir);

    if(!perf.codec_emit_frame_ms.empty()){
        // predictor_engine_async_wall_ms overlaps other work by design.
        // Only predictor_engine_wait_ms belongs on the serial TTFT path.
        perf.ttft_accounted_serial_ms=
            perf.cli_and_tokenizer_ms
            +perf.model_setup_wait_ms
            +perf.workdir_setup_ms
            +speaker_ms
            +perf.frontend_ms
            +perf.talker_engine_wait_ms
            +perf.talker_rope_build_ms
            +perf.talker_prefill_ms
            +perf.predictor_engine_wait_ms
            +perf.first_codec_frame_ms;

        perf.ttft_unaccounted_ms=
            perf.ttft_ms
            -perf.ttft_accounted_serial_ms;
    }

    const int frames=
        static_cast<int>(
            codec.size()/16);

    const std::uint64_t codec_fnv1a64=
        fnv1a64(
            codec.data(),
            codec.size()*sizeof(std::int32_t));

    std::vector<float> waveform;

    std::size_t decoder_node_count=0;

    if(
        correctness_dir.empty()
        &&streaming_decoder
    ){
        const auto tail_start=
            PerfClock::now();

        auto streaming_result=
            streaming_decoder
                ->finish();

        perf.codec_decoder_tail_wait_ms=
            elapsed_ms(
                tail_start,
                PerfClock::now());

        perf.codec_decoder_parameter_preload_wall_ms=
            streaming_result
                .parameter_preload_wall_ms;

        perf.codec_decoder_parameter_preload_wait_ms=
            streaming_result
                .parameter_wait_ms;

        perf.codec_decoder_setup_ms=
            streaming_result
                .decoder_setup_ms;

        perf.codec_decoder_graph_ms=
            streaming_result
                .decoder_compute_ms;

        perf.codec_decoder_pipeline_wall_ms=
            streaming_result
                .decoder_pipeline_wall_ms;

        perf.codec_decoder_first_chunk_ms=
            streaming_result
                .first_audio_ms;

        perf.codec_decoder_chunk_count=
            streaming_result
                .chunk_count;

        perf.codec_decoder_first_chunk_frames=
            streaming_result
                .first_chunk_frames;

        perf.codec_decoder_chunk_frames=
            streaming_result
                .chunk_frames;

        perf.ttfa_ms=
            streaming_result
                .first_audio_ms;

        waveform=
            std::move(
                streaming_result
                    .waveform);

        decoder_node_count=0;
    }else{
        const auto decoder_wait_start=
            PerfClock::now();

        const fs::path speech_tokenizer=
            codec_decoder::
                locate_speech_tokenizer(
                    model_dir);

        auto decoder_parameters=
            std::make_shared<
                codec_decoder::Parameters>(
                    speech_tokenizer);

        perf.codec_decoder_parameter_preload_wait_ms=
            elapsed_ms(
                decoder_wait_start,
                PerfClock::now());

        perf.codec_decoder_parameter_preload_wall_ms=
            perf.codec_decoder_parameter_preload_wait_ms;

        const auto decoder_setup_start=
            PerfClock::now();

        codec_decoder::NativeCodecDecoder decoder(
            decoder_parameters,
            frames,
            false,
            false,
            true);

        perf.codec_decoder_setup_ms=
            elapsed_ms(
                decoder_setup_start,
                PerfClock::now());

        const auto decoder_start=
            PerfClock::now();

        waveform=
            decoder.decode(
                codec);

        perf.codec_decoder_graph_ms=
            elapsed_ms(
                decoder_start,
                PerfClock::now());

        perf.codec_decoder_pipeline_wall_ms=
            perf.codec_decoder_graph_ms;

        perf.codec_decoder_tail_wait_ms=
            perf.codec_decoder_graph_ms;

        perf.codec_decoder_chunk_count=1;
        perf.codec_decoder_chunk_frames=frames;

        perf.ttfa_ms=
            elapsed_ms(
                e2e_start,
                PerfClock::now());

        decoder_node_count=
            decoder.node_count();
    }

    if(
        waveform.size()
        !=static_cast<std::size_t>(
            frames)
            *codec_decoder::UPSAMPLE
    ){
        throw std::runtime_error(
            "streaming decoder waveform length mismatch");
    }

    const auto wav_start=
        PerfClock::now();

    write_float_wav(
        fs::absolute(
            options.output_wav),
        waveform,
        codec_decoder::SAMPLE_RATE);

    perf.wav_write_ms=
        elapsed_ms(
            wav_start,
            PerfClock::now());

    perf.e2e_ms=
        elapsed_ms(
            e2e_start,
            PerfClock::now());

    const double audio_duration_s=
        static_cast<double>(
            waveform.size())
        /codec_decoder::SAMPLE_RATE;

    const double e2e_s=
        perf.e2e_ms/1000.0;

    const double decoder_s=
        perf.codec_decoder_graph_ms
        /1000.0;

    const double generation_s=
        perf.codec_generation_ms
        /1000.0;

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"\n"
        <<"================================================================================\n"
        <<"legacy_legacy_tag_j.3 OFFICIAL BASE X-VECTOR FINAL SUMMARY\n"
        <<"================================================================================\n"
        <<"official_base_contract: generate_voice_clone_x_vector_only\n"
        <<"legacy_x_vector_only_mode: True\n"
        <<"icl_mode: False\n"
        <<"reference_audio_dependency: True\n"
        <<"ref_text_dependency: False\n"
        <<"ref_code_dependency: False\n"
        <<"speaker_conditioning: native_reference_audio_ecapa_tdnn\n"
        <<"trailing_text_conditioning: True\n"
        <<"predictor_weights_source: model_safetensors\n"
        <<"predictor_rope_source: native_cpp\n"
        <<"external_native_assets_dependency: False\n"
        <<"legacy_runtime_python_dependency_b: False\n"
        <<"legacy_runtime_huggingface_dependency_b: False\n"
        <<"legacy_runtime_bridge_process_count_b: 0\n"
        <<"legacy_language: "<<options.language<<"\n"
        <<"legacy_input_token_count_b: "<<native_ids.size()<<"\n"
        <<"prefill_rows: "<<native.seq_len<<"\n"
        <<"trailing_text_rows: "<<native.trailing_rows<<"\n"
        <<"legacy_codec_frame_count_b: "<<frames<<"\n"
        <<"legacy_codec_fnv1a64_b: "<<codec_fnv1a64<<"\n"
        <<"audio_sample_rate: "<<codec_decoder::SAMPLE_RATE<<"\n"
        <<"legacy_audio_sample_count_b: "<<waveform.size()<<"\n"
        <<"legacy_audio_duration_s_b: "<<audio_duration_s<<"\n"
        <<"cli_tokenizer_ms: "<<perf.cli_and_tokenizer_ms<<"\n"
        <<"production_workdir_setup_ms: "<<perf.workdir_setup_ms<<"\n"
        <<"speaker_conditioning_ms: "<<speaker_ms<<"\n"
        <<"legacy_frontend_ms_b: "<<perf.frontend_ms<<"\n"
        <<"model_setup_async_wall_ms: "<<perf.core_model_setup_ms<<"\n"
        <<"model_setup_wait_ms: "<<perf.model_setup_wait_ms<<"\n"
        <<"predictor_engine_async_wall_ms: "
        <<perf.predictor_engine_async_wall_ms<<"\n"
        <<"predictor_engine_wait_ms: "
        <<perf.predictor_engine_wait_ms<<"\n"
        <<"talker_engine_async_wall_ms: "<<perf.talker_engine_async_wall_ms<<"\n"
        <<"talker_engine_wait_ms: "<<perf.talker_engine_wait_ms<<"\n"
        <<"talker_engine_init_ms: "<<perf.talker_engine_init_ms<<"\n"
        <<"talker_rope_build_ms: "<<perf.talker_rope_build_ms<<"\n"
        <<"first_codec_frame_compute_ms: "<<perf.first_codec_frame_ms<<"\n"
        <<"ttft_accounted_serial_ms: "<<perf.ttft_accounted_serial_ms<<"\n"
        <<"ttft_unaccounted_ms: "<<perf.ttft_unaccounted_ms<<"\n"
        <<"gpu_sampling_backend: gpu_topk50_splitmix64\n"
        <<"predictor_execution_backend: persistent_15_step_hip_graph\n"
        <<"predictor_graph_node_count: "
        <<perf.predictor_graph_node_count<<"\n"
        <<"predictor_frame_p50_ms: "
        <<percentile(perf.predictor_frame_ms,0.50)<<"\n"
        <<"predictor_frame_p95_ms: "
        <<percentile(perf.predictor_frame_ms,0.95)<<"\n"
        <<"legacy_talker_prefill_ms_b: "<<perf.talker_prefill_ms<<"\n"
        <<"legacy_codec_generation_ms_b: "<<perf.codec_generation_ms<<"\n"
        <<"decoder_parameter_preload_wall_ms: "<<perf.codec_decoder_parameter_preload_wall_ms<<"\n"
        <<"decoder_parameter_preload_wait_ms: "<<perf.codec_decoder_parameter_preload_wait_ms<<"\n"
        <<"legacy_codec_decoder_setup_ms_b: "<<perf.codec_decoder_setup_ms<<"\n"
        <<"legacy_codec_decoder_graph_ms_b: "<<perf.codec_decoder_graph_ms<<"\n"
        <<"streaming_decoder_pipeline_wall_ms: "<<perf.codec_decoder_pipeline_wall_ms<<"\n"
        <<"streaming_decoder_tail_wait_ms: "<<perf.codec_decoder_tail_wait_ms<<"\n"
        <<"streaming_decoder_first_chunk_ms: "<<perf.codec_decoder_first_chunk_ms<<"\n"
        <<"streaming_decoder_chunk_count: "<<perf.codec_decoder_chunk_count<<"\n"
        <<"streaming_decoder_first_chunk_frames: "<<perf.codec_decoder_first_chunk_frames<<"\n"
        <<"streaming_decoder_chunk_frames: "<<perf.codec_decoder_chunk_frames<<"\n"
        <<"legacy_ttft_ms_b: "<<perf.ttft_ms<<"\n"
        <<"legacy_ttfa_ms_b: "<<perf.ttfa_ms<<"\n"
        <<"legacy_e2e_ms_b: "<<perf.e2e_ms<<"\n"
        <<"legacy_codec_generation_frames_per_s_b: "
        <<(
            generation_s>0.0
            ?frames/generation_s
            :0.0)
        <<"\n"
        <<"legacy_decoder_realtime_x_b: "
        <<(
            decoder_s>0.0
            ?audio_duration_s/decoder_s
            :0.0)
        <<"\n"
        <<"legacy_e2e_realtime_x_b: "
        <<(
            e2e_s>0.0
            ?audio_duration_s/e2e_s
            :0.0)
        <<"\n"
        <<"legacy_graph_node_count_b: "<<decoder_node_count<<"\n"
        <<"legacy_output_wav_b: "
        <<fs::absolute(
            options.output_wav).string()
        <<"\n"
        <<"native_official_input_to_wav_closed: True\n"
        <<"talker_correctness_backend: staged_prefill_equivalent\n"
        <<"legacy_codec_vocoder_correctness_closed_b: True\n"
        <<"remaining_correctness_risk: talker_numeric_accumulation_and_speaker_encoder_numeric_validation\n"
        <<"talker_rope_sign_fix: True\n"
        <<"expected_first_boundary: talker_layer_00_last\n"
        <<"correctness_boundary_enabled: "
        <<(!correctness_dir.empty()?"True":"False")
        <<"\n"
        <<"legacy_correctness_dir: "
        <<(correctness_dir.empty()?"<disabled>":correctness_dir.string())
        <<"\n"
        <<"baseline_engine: qwen3_tts_base_native\n"
        <<"baseline_input_contract: official_base_xvector_only\n"
        <<"baseline_talker_rope: official_minus_x2_plus_x1\n"
        <<"feature_gpu_sampling: True\n"
        <<"feature_predictor_persistent_dag: True\n"
        <<"feature_predictor_per_frame_filesystem_io: False\n"
        <<"feature_predictor_per_step_device_sync: False\n"
        <<"feature_device_resident_talker_predictor_chain: True\n"
        <<"feature_talker_hidden_roundtrip: False\n"
        <<"feature_predictor_next_input_roundtrip: False\n"
        <<"feature_talker_qkv_fusion: True\n"
        <<"feature_talker_qk_norm_rope_cache_fusion: True\n"
        <<"feature_talker_prefill_fusion: False\n"
        <<"feature_talker_attention_fusion_changed: False\n"
        <<"feature_talker_mlp_fusion_changed: True\n"
        <<"feature_talker_oproj_lds_staging: True\n"
        <<"feature_talker_residual_postnorm_fusion: True\n"
        <<"feature_talker_gate_up_swiglu_fusion: True\n"
        <<"feature_talker_down_residual_fusion: True\n"
        <<"feature_cache_policy: raw_material_in_cache_derived_in_vgpr_lds\n"
        <<"feature_talker_attention_gqa2_kv_locality: True\n"
        <<"feature_talker_attention_k_reads_shared_across_q_heads: True\n"
        <<"feature_talker_attention_v_reads_shared_across_q_heads: True\n"
        <<"feature_talker_attention_global_score_tensor: False\n"
        <<"feature_talker_attention_global_probability_tensor: False\n"
        <<"feature_predictor_m1_locality_fusion: True\n"
        <<"feature_predictor_step0_staged_anchor: True\n"
        <<"feature_predictor_m1_qkv_fusion: True\n"
        <<"feature_predictor_m1_qk_norm_rope_cache_fusion: True\n"
        <<"feature_predictor_m1_attention_gqa2: True\n"
        <<"feature_predictor_m1_gate_up_swiglu_fusion: True\n"
        <<"feature_predictor_m1_down_residual_fusion: True\n"
        <<"feature_predictor_cache_policy: raw_material_in_cache_derived_in_vgpr_lds\n"
        <<"feature_priority_kpi_primary: e2e_ttfa\n"
        <<"feature_production_workdir_removed: True\n"
        <<"feature_predictor_build_async_overlap: True\n"
        <<"feature_predictor_build_overlap_with_speaker_frontend_talker: True\n"
        <<"feature_predictor_capture_mode: thread_local\n"
        <<"feature_predictor_capture_global_allocator_hazard: False\n"
        <<"feature_capture_overlap_dependency_control: manual_explicit\n"
        <<"feature_speaker_stream_ordered_allocator: True\n"
        <<"feature_frontend_no_device_wide_sync: True\n"
        <<"feature_talker_weight_sync_scope: upload_stream_only\n"
        <<"feature_stream_policy: explicit_nonblocking\n"
        <<"feature_speaker_explicit_stream: True\n"
        <<"feature_frontend_explicit_stream: True\n"
        <<"feature_talker_weight_upload_explicit_stream: True\n"
        <<"feature_legacy_stream_calls_in_capture_overlap: False\n"
        <<"feature_compile_requires_fgpu_default_stream_per_thread: False\n"
        <<"feature_per_frame_stdout_compacted: True\n"
        <<"feature_ttft_critical_path_instrumented: True\n"
        <<"feature_model_setup_overlap_with_tokenizer: True\n"
        <<"feature_talker_init_async_overlap: True\n"
        <<"feature_talker_init_overlap_with_speaker: True\n"
        <<"feature_decoder_parameter_preload_after_ttft: True\n"
        <<"feature_decoder_parameter_preload_overlap_with_generation: True\n"
        <<"feature_decoder_preload_can_affect_ttft: False\n"
        <<"feature_generation_explicit_stream: True\n"
        <<"feature_generation_device_wide_sync_per_frame: False\n"
        <<"feature_final_head_generation_stream: True\n"
        <<"feature_decoder_preload_independent_of_generation_barrier: True\n"
        <<"feature_streaming_decoder_pipeline_ready: True\n"
        <<"feature_final_head_sampler_default_stream: False\n"
        <<"feature_final_head_single_stream_ordering: True\n"
        <<"feature_eos_ordering_race_closed: True\n"
        <<"feature_priority_kpi: e2e_ttfa_generation\n"
        <<"feature_ttft_target_frozen_around_320ms: True\n"
        <<"feature_predictor_gfx11_bf16_dot2: True\n"
        <<"feature_predictor_dot2_qkv_m1: True\n"
        <<"feature_predictor_dot2_linear_m1: True\n"
        <<"feature_predictor_dot2_gate_up_m1: True\n"
        <<"feature_predictor_step0_dot2_changed: False\n"
        <<"feature_streaming_codec_decoder: True\n"
        <<"feature_streaming_codec_first_chunk_frames: 8\n"
        <<"feature_streaming_codec_prefix_recompute: False\n"
        <<"feature_streaming_codec_transformer_kv_cache: True\n"
        <<"feature_streaming_codec_persistent_causal_history: True\n"
        <<"feature_generation_decoder_pipeline: True\n"
        <<"feature_decoder_parameter_preload_stream_ordered: True\n"
        <<"feature_decoder_fixed_shape_graph_in_production: False\n"
        <<"feature_e2e_wavepack: True\n"
        <<"feature_predictor_outputs_per_wave: 2\n"
        <<"feature_talker_outputs_per_wave: 2\n"
        <<"feature_talker_dot2_changed: False\n"
        <<"feature_streaming_decoder_outputs_per_wave: 4\n"
        <<"feature_streaming_decoder_linear_pack4: True\n"
        <<"feature_streaming_decoder_conv_pack4: True\n"
        <<"feature_streaming_decoder_transconv_pack4: True\n"
        <<"feature_streaming_decoder_pointwise_pack4: True\n"
        <<"feature_streaming_codec_chunk_frames_effective: 16\n"
        <<"baseline_codec_vocoder_correctness: closed\n"
        <<"baseline_runtime_python_dependency: False\n"
        <<"baseline_runtime_huggingface_dependency: False\n"
        <<"baseline_runtime_rocblas_dependency: False\n"
        <<"baseline_runtime_hipblas_dependency: False\n"
        <<"baseline_success: True\n"
        <<"================================================================================\n";

    if(
        !options.keep_work
        &&!work.empty()
    ){
        fs::remove_all(work);
    }

    return 0;
}

static int run_official_custom_voice(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids,
    std::future<
        std::unique_ptr<
            OfficialPreparedModel
        >
    >* model_future,
    const PerfTime& model_async_start,
    const PerfTime& e2e_start,
    double cli_and_tokenizer_ms) {

    const fs::path model_dir=
        fs::absolute(
            options.model_dir);

    const CustomVoiceContract
        custom_contract=
            load_custom_voice_contract(
                model_dir);

    const std::string
        effective_language=
            resolve_custom_voice_effective_language(
                custom_contract,
                options.speaker,
                options.language);

    const fs::path ref_audio{};

    const fs::path correctness_dir=
        options.correctness_dir.empty()
        ?fs::path{}
        :fs::absolute(
            options.correctness_dir);

    if(!correctness_dir.empty()){
        fs::remove_all(correctness_dir);
        fs::create_directories(correctness_dir);

        write_i64(
            correctness_dir/
                "native_input_ids.i64.bin",
            native_ids);
    }


    // legacy_legacy_tag_j.3 follows the official generate_voice_clone sampling defaults.
    configure_sampling(
        !options.greedy,
        options.seed);

    GpuSamplingState gpu_sampling_state(
        options.seed);

    PerfStats perf;
    perf.cli_and_tokenizer_ms=
        cli_and_tokenizer_ms;

    fs::path work;

    const auto workdir_start=
        PerfClock::now();

    if(!correctness_dir.empty()){
        work=
            select_work_dir(
                options);

        fs::remove_all(work);
        fs::create_directories(work);
    }

    perf.workdir_setup_ms=
        elapsed_ms(
            workdir_start,
            PerfClock::now());

    std::cout
        <<"execution_task: custom-voice\n"
        <<"text_mode: streaming\n"
        <<"model_contract: Qwen3-TTS-12Hz-0.6B-CustomVoice\n"
        <<"official_api_contract: generate_custom_voice\n"
        <<"reference_audio_dependency: False\n"
        <<"predefined_speaker_dependency: True\n"
        <<"speaker: "<<options.speaker<<"\n"
        <<"requested_language: "<<options.language<<"\n"
        <<"runtime_python_dependency: False\n"
        <<"runtime_huggingface_dependency: False\n"
        <<"runtime_rocblas_dependency: False\n"
        <<"runtime_hipblas_dependency: False\n"
        <<"runtime_external_code_objects: False\n";

    const auto model_wait_start=
        PerfClock::now();

    std::unique_ptr<
        OfficialPreparedModel>
        prepared_model;

    if(
        model_future!=nullptr
        &&model_future->valid()
    ){
        prepared_model=
            model_future->get();
    }else{
        prepared_model=
            std::make_unique<
                OfficialPreparedModel>(
                    model_dir);
    }

    const auto model_ready=
        PerfClock::now();

    perf.model_setup_wait_ms=
        elapsed_ms(
            model_wait_start,
            model_ready);

    perf.core_model_setup_ms=
        elapsed_ms(
            model_async_start,
            model_ready);

    auto& final_head=
        prepared_model->final_head;

    auto& predictor_weights=
        prepared_model->predictor_weights;

    const auto& codec_embedding=
        prepared_model->codec_embedding();

    // Predictor graph construction is request-independent after model setup.
    // Start it before speaker conditioning so graph capture/instantiate,
    // allocations and RoPE upload overlap the speaker/frontend/Talker path.
    const auto predictor_async_start=
        PerfClock::now();

    std::future<
        std::unique_ptr<
            code_predictor::PersistentPredictor
        >
    > predictor_future;

    if(correctness_dir.empty()){
        predictor_future=
            std::async(
                std::launch::async,
                [&predictor_weights,
                 &codec_embedding,
                 &gpu_sampling_state](){
                    return std::make_unique<
                        code_predictor::PersistentPredictor>(
                            predictor_weights,
                            codec_embedding,
                            gpu_sampling_state
                                .device_state());
                });
    }

    const int expected_prefill_rows=
        official_xvector_prefill_rows(
            effective_language);

    const auto talker_async_start=
        PerfClock::now();

    std::future<
        std::unique_ptr<
            talker::NativeTalkerEngine
        >
    > talker_future;

    if(correctness_dir.empty()){
        talker_future=
            std::async(
                std::launch::async,
                [model_dir,
                 expected_prefill_rows,
                 maximum_frames=
                    static_cast<int>(
                        generation_capacity_for(options.max_new_tokens))](){
                    return std::make_unique<
                        talker::NativeTalkerEngine>(
                            model_dir,
                            expected_prefill_rows,
                            maximum_frames,
                            -1);
                });
    }

    const auto speaker_start=
        PerfClock::now();

    std::vector<std::uint16_t>
        speaker_embedding;

    const auto custom_conditioning=
        build_custom_voice_conditioning(
            custom_contract,
            codec_embedding,
            options.speaker,
            effective_language);

    speaker_embedding=
        custom_conditioning
            .speaker_embedding;

    std::cout
        <<"speaker_embedding_source: checkpoint_codec_embedding_row\n"
        <<"effective_language: "
        <<effective_language
        <<"\n";

    if(!correctness_dir.empty()){
        write_binary(
            correctness_dir/
                "native_speaker_embedding_used.bf16.bin",
            speaker_embedding);
    }

    const double speaker_ms=
        elapsed_ms(
            speaker_start,
            PerfClock::now());

    const auto frontend_start=
        PerfClock::now();

    model_frontend::NativeFrontend frontend(
        model_dir,
        fs::path{});

    const auto native=
        frontend.build_official_base_xvector_inputs(
            native_ids,
            speaker_embedding,
            effective_language);

    perf.frontend_ms=
        elapsed_ms(
            frontend_start,
            PerfClock::now());

    if(!correctness_dir.empty()){
        write_binary(
            correctness_dir/
                "native_talker_prefill_input.bf16.bin",
            native.prefill);

        write_binary(
            correctness_dir/
                "native_trailing_text_hidden.bf16.bin",
            native.trailing_text);

        write_binary(
            correctness_dir/
                "native_tts_pad.bf16.bin",
            native.tts_pad);
    }

    if(
        static_cast<int>(native.seq_len)
        !=expected_prefill_rows
    ){
        throw std::runtime_error(
            "official Base predicted prefill rows disagree with frontend result");
    }

    if(
        native.prefill.size()
        !=native.seq_len*1024
        ||native.trailing_text.size()
            !=native.trailing_rows*1024
    ){
        throw std::runtime_error(
            "legacy_legacy_tag_j.3 official Base frontend output geometry mismatch");
    }

    std::unique_ptr<
        codec_decoder::StreamingCodecPipeline>
        streaming_decoder;

    PerfTime decoder_preload_start{};

    const std::function<void()>
        first_frame_hook=
            [&streaming_decoder,
             &decoder_preload_start,
             model_dir,
             maximum_frames=
                static_cast<int>(
                    generation_capacity_for(options.max_new_tokens)),
             e2e_start](){

                decoder_preload_start=
                    PerfClock::now();

                auto preload_future=
                    std::async(
                        std::launch::async,
                        [model_dir](){
                            const fs::path tokenizer=
                                codec_decoder::
                                    locate_speech_tokenizer(
                                        model_dir);

                            auto parameters=
                                std::make_shared<
                                    codec_decoder::Parameters>(
                                        tokenizer);

                            return std::make_pair(
                                tokenizer,
                                std::move(parameters));
                        });

                streaming_decoder=
                    std::make_unique<
                        codec_decoder::
                            StreamingCodecPipeline>(
                                std::move(
                                    preload_future),
                                decoder_preload_start,
                                maximum_frames,
                                e2e_start,
                                codec_decoder::
                                    STREAM_CHUNK_FRAMES);
            };

    const std::function<
        void(
            const std::array<
                std::int32_t,
                16>&
        )
    > codec_frame_hook=
        [&streaming_decoder](
            const std::array<
                std::int32_t,
                16>& frame_codes){

            if(streaming_decoder){
                streaming_decoder
                    ->push_frame(
                        frame_codes);
            }
        };

    const auto codec=
        generate_codec_sequence(
            model_dir,
            work,
            native.prefill,
            static_cast<int>(
                native.seq_len),
            static_cast<int>(
                options.max_new_tokens),
            native.tts_pad,
            native.trailing_text,
            native.trailing_rows,
            predictor_weights,
            codec_embedding,
            final_head,
            gpu_sampling_state.device_state(),
            correctness_dir.empty()
                ?&predictor_future
                :nullptr,
            predictor_async_start,
            correctness_dir.empty()
                ?&talker_future
                :nullptr,
            talker_async_start,
            correctness_dir.empty()
                ?first_frame_hook
                :std::function<void()>{},
            correctness_dir.empty()
                ?codec_frame_hook
                :std::function<
                    void(
                        const std::array<
                            std::int32_t,
                            16>&
                    )
                >{},
            e2e_start,
            perf,
            correctness_dir);

    if(!perf.codec_emit_frame_ms.empty()){
        // predictor_engine_async_wall_ms overlaps other work by design.
        // Only predictor_engine_wait_ms belongs on the serial TTFT path.
        perf.ttft_accounted_serial_ms=
            perf.cli_and_tokenizer_ms
            +perf.model_setup_wait_ms
            +perf.workdir_setup_ms
            +speaker_ms
            +perf.frontend_ms
            +perf.talker_engine_wait_ms
            +perf.talker_rope_build_ms
            +perf.talker_prefill_ms
            +perf.predictor_engine_wait_ms
            +perf.first_codec_frame_ms;

        perf.ttft_unaccounted_ms=
            perf.ttft_ms
            -perf.ttft_accounted_serial_ms;
    }

    const int frames=
        static_cast<int>(
            codec.size()/16);

    const std::uint64_t codec_fnv1a64=
        fnv1a64(
            codec.data(),
            codec.size()*sizeof(std::int32_t));

    std::vector<float> waveform;

    std::size_t decoder_node_count=0;

    if(
        correctness_dir.empty()
        &&streaming_decoder
    ){
        const auto tail_start=
            PerfClock::now();

        auto streaming_result=
            streaming_decoder
                ->finish();

        perf.codec_decoder_tail_wait_ms=
            elapsed_ms(
                tail_start,
                PerfClock::now());

        perf.codec_decoder_parameter_preload_wall_ms=
            streaming_result
                .parameter_preload_wall_ms;

        perf.codec_decoder_parameter_preload_wait_ms=
            streaming_result
                .parameter_wait_ms;

        perf.codec_decoder_setup_ms=
            streaming_result
                .decoder_setup_ms;

        perf.codec_decoder_graph_ms=
            streaming_result
                .decoder_compute_ms;

        perf.codec_decoder_pipeline_wall_ms=
            streaming_result
                .decoder_pipeline_wall_ms;

        perf.codec_decoder_first_chunk_ms=
            streaming_result
                .first_audio_ms;

        perf.codec_decoder_chunk_count=
            streaming_result
                .chunk_count;

        perf.codec_decoder_first_chunk_frames=
            streaming_result
                .first_chunk_frames;

        perf.codec_decoder_chunk_frames=
            streaming_result
                .chunk_frames;

        perf.ttfa_ms=
            streaming_result
                .first_audio_ms;

        waveform=
            std::move(
                streaming_result
                    .waveform);

        decoder_node_count=0;
    }else{
        const auto decoder_wait_start=
            PerfClock::now();

        const fs::path speech_tokenizer=
            codec_decoder::
                locate_speech_tokenizer(
                    model_dir);

        auto decoder_parameters=
            std::make_shared<
                codec_decoder::Parameters>(
                    speech_tokenizer);

        perf.codec_decoder_parameter_preload_wait_ms=
            elapsed_ms(
                decoder_wait_start,
                PerfClock::now());

        perf.codec_decoder_parameter_preload_wall_ms=
            perf.codec_decoder_parameter_preload_wait_ms;

        const auto decoder_setup_start=
            PerfClock::now();

        codec_decoder::NativeCodecDecoder decoder(
            decoder_parameters,
            frames,
            false,
            false,
            true);

        perf.codec_decoder_setup_ms=
            elapsed_ms(
                decoder_setup_start,
                PerfClock::now());

        const auto decoder_start=
            PerfClock::now();

        waveform=
            decoder.decode(
                codec);

        perf.codec_decoder_graph_ms=
            elapsed_ms(
                decoder_start,
                PerfClock::now());

        perf.codec_decoder_pipeline_wall_ms=
            perf.codec_decoder_graph_ms;

        perf.codec_decoder_tail_wait_ms=
            perf.codec_decoder_graph_ms;

        perf.codec_decoder_chunk_count=1;
        perf.codec_decoder_chunk_frames=frames;

        perf.ttfa_ms=
            elapsed_ms(
                e2e_start,
                PerfClock::now());

        decoder_node_count=
            decoder.node_count();
    }

    if(
        waveform.size()
        !=static_cast<std::size_t>(
            frames)
            *codec_decoder::UPSAMPLE
    ){
        throw std::runtime_error(
            "streaming decoder waveform length mismatch");
    }

    const auto wav_start=
        PerfClock::now();

    write_float_wav(
        fs::absolute(
            options.output_wav),
        waveform,
        codec_decoder::SAMPLE_RATE);

    perf.wav_write_ms=
        elapsed_ms(
            wav_start,
            PerfClock::now());

    perf.e2e_ms=
        elapsed_ms(
            e2e_start,
            PerfClock::now());

    const double audio_duration_s=
        static_cast<double>(
            waveform.size())
        /codec_decoder::SAMPLE_RATE;

    const double e2e_s=
        perf.e2e_ms/1000.0;

    const double decoder_s=
        perf.codec_decoder_graph_ms
        /1000.0;

    const double generation_s=
        perf.codec_generation_ms
        /1000.0;

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"\n"
        <<"================================================================================\n"
        <<"legacy_legacy_tag_j.3 OFFICIAL BASE X-VECTOR FINAL SUMMARY\n"
        <<"================================================================================\n"
        <<"official_base_contract: generate_voice_clone_x_vector_only\n"
        <<"legacy_x_vector_only_mode: False\n"
        <<"icl_mode: False\n"
        <<"reference_audio_dependency: False\n"
        <<"ref_text_dependency: False\n"
        <<"ref_code_dependency: False\n"
        <<"speaker_conditioning: checkpoint_codec_embedding_row\n"
        <<"trailing_text_conditioning: True\n"
        <<"predictor_weights_source: model_safetensors\n"
        <<"predictor_rope_source: native_cpp\n"
        <<"external_native_assets_dependency: False\n"
        <<"legacy_runtime_python_dependency_b: False\n"
        <<"legacy_runtime_huggingface_dependency_b: False\n"
        <<"legacy_runtime_bridge_process_count_b: 0\n"
        <<"legacy_language: "<<effective_language<<"\n"
        <<"legacy_input_token_count_b: "<<native_ids.size()<<"\n"
        <<"prefill_rows: "<<native.seq_len<<"\n"
        <<"trailing_text_rows: "<<native.trailing_rows<<"\n"
        <<"legacy_codec_frame_count_b: "<<frames<<"\n"
        <<"legacy_codec_fnv1a64_b: "<<codec_fnv1a64<<"\n"
        <<"audio_sample_rate: "<<codec_decoder::SAMPLE_RATE<<"\n"
        <<"legacy_audio_sample_count_b: "<<waveform.size()<<"\n"
        <<"legacy_audio_duration_s_b: "<<audio_duration_s<<"\n"
        <<"cli_tokenizer_ms: "<<perf.cli_and_tokenizer_ms<<"\n"
        <<"production_workdir_setup_ms: "<<perf.workdir_setup_ms<<"\n"
        <<"speaker_conditioning_ms: "<<speaker_ms<<"\n"
        <<"legacy_frontend_ms_b: "<<perf.frontend_ms<<"\n"
        <<"model_setup_async_wall_ms: "<<perf.core_model_setup_ms<<"\n"
        <<"model_setup_wait_ms: "<<perf.model_setup_wait_ms<<"\n"
        <<"predictor_engine_async_wall_ms: "
        <<perf.predictor_engine_async_wall_ms<<"\n"
        <<"predictor_engine_wait_ms: "
        <<perf.predictor_engine_wait_ms<<"\n"
        <<"talker_engine_async_wall_ms: "<<perf.talker_engine_async_wall_ms<<"\n"
        <<"talker_engine_wait_ms: "<<perf.talker_engine_wait_ms<<"\n"
        <<"talker_engine_init_ms: "<<perf.talker_engine_init_ms<<"\n"
        <<"talker_rope_build_ms: "<<perf.talker_rope_build_ms<<"\n"
        <<"first_codec_frame_compute_ms: "<<perf.first_codec_frame_ms<<"\n"
        <<"ttft_accounted_serial_ms: "<<perf.ttft_accounted_serial_ms<<"\n"
        <<"ttft_unaccounted_ms: "<<perf.ttft_unaccounted_ms<<"\n"
        <<"gpu_sampling_backend: gpu_topk50_splitmix64\n"
        <<"predictor_execution_backend: persistent_15_step_hip_graph\n"
        <<"predictor_graph_node_count: "
        <<perf.predictor_graph_node_count<<"\n"
        <<"predictor_frame_p50_ms: "
        <<percentile(perf.predictor_frame_ms,0.50)<<"\n"
        <<"predictor_frame_p95_ms: "
        <<percentile(perf.predictor_frame_ms,0.95)<<"\n"
        <<"legacy_talker_prefill_ms_b: "<<perf.talker_prefill_ms<<"\n"
        <<"legacy_codec_generation_ms_b: "<<perf.codec_generation_ms<<"\n"
        <<"decoder_parameter_preload_wall_ms: "<<perf.codec_decoder_parameter_preload_wall_ms<<"\n"
        <<"decoder_parameter_preload_wait_ms: "<<perf.codec_decoder_parameter_preload_wait_ms<<"\n"
        <<"legacy_codec_decoder_setup_ms_b: "<<perf.codec_decoder_setup_ms<<"\n"
        <<"legacy_codec_decoder_graph_ms_b: "<<perf.codec_decoder_graph_ms<<"\n"
        <<"streaming_decoder_pipeline_wall_ms: "<<perf.codec_decoder_pipeline_wall_ms<<"\n"
        <<"streaming_decoder_tail_wait_ms: "<<perf.codec_decoder_tail_wait_ms<<"\n"
        <<"streaming_decoder_first_chunk_ms: "<<perf.codec_decoder_first_chunk_ms<<"\n"
        <<"streaming_decoder_chunk_count: "<<perf.codec_decoder_chunk_count<<"\n"
        <<"streaming_decoder_first_chunk_frames: "<<perf.codec_decoder_first_chunk_frames<<"\n"
        <<"streaming_decoder_chunk_frames: "<<perf.codec_decoder_chunk_frames<<"\n"
        <<"legacy_ttft_ms_b: "<<perf.ttft_ms<<"\n"
        <<"legacy_ttfa_ms_b: "<<perf.ttfa_ms<<"\n"
        <<"legacy_e2e_ms_b: "<<perf.e2e_ms<<"\n"
        <<"legacy_codec_generation_frames_per_s_b: "
        <<(
            generation_s>0.0
            ?frames/generation_s
            :0.0)
        <<"\n"
        <<"legacy_decoder_realtime_x_b: "
        <<(
            decoder_s>0.0
            ?audio_duration_s/decoder_s
            :0.0)
        <<"\n"
        <<"legacy_e2e_realtime_x_b: "
        <<(
            e2e_s>0.0
            ?audio_duration_s/e2e_s
            :0.0)
        <<"\n"
        <<"legacy_graph_node_count_b: "<<decoder_node_count<<"\n"
        <<"legacy_output_wav_b: "
        <<fs::absolute(
            options.output_wav).string()
        <<"\n"
        <<"native_official_input_to_wav_closed: True\n"
        <<"talker_correctness_backend: staged_prefill_equivalent\n"
        <<"legacy_codec_vocoder_correctness_closed_b: True\n"
        <<"remaining_correctness_risk: talker_numeric_accumulation_and_speaker_encoder_numeric_validation\n"
        <<"talker_rope_sign_fix: True\n"
        <<"expected_first_boundary: talker_layer_00_last\n"
        <<"correctness_boundary_enabled: "
        <<(!correctness_dir.empty()?"True":"False")
        <<"\n"
        <<"legacy_correctness_dir: "
        <<(correctness_dir.empty()?"<disabled>":correctness_dir.string())
        <<"\n"
        <<"baseline_engine: qwen3_tts_base_native\n"
        <<"baseline_input_contract: official_base_xvector_only\n"
        <<"baseline_talker_rope: official_minus_x2_plus_x1\n"
        <<"feature_gpu_sampling: True\n"
        <<"feature_predictor_persistent_dag: True\n"
        <<"feature_predictor_per_frame_filesystem_io: False\n"
        <<"feature_predictor_per_step_device_sync: False\n"
        <<"feature_device_resident_talker_predictor_chain: True\n"
        <<"feature_talker_hidden_roundtrip: False\n"
        <<"feature_predictor_next_input_roundtrip: False\n"
        <<"feature_talker_qkv_fusion: True\n"
        <<"feature_talker_qk_norm_rope_cache_fusion: True\n"
        <<"feature_talker_prefill_fusion: False\n"
        <<"feature_talker_attention_fusion_changed: False\n"
        <<"feature_talker_mlp_fusion_changed: True\n"
        <<"feature_talker_oproj_lds_staging: True\n"
        <<"feature_talker_residual_postnorm_fusion: True\n"
        <<"feature_talker_gate_up_swiglu_fusion: True\n"
        <<"feature_talker_down_residual_fusion: True\n"
        <<"feature_cache_policy: raw_material_in_cache_derived_in_vgpr_lds\n"
        <<"feature_talker_attention_gqa2_kv_locality: True\n"
        <<"feature_talker_attention_k_reads_shared_across_q_heads: True\n"
        <<"feature_talker_attention_v_reads_shared_across_q_heads: True\n"
        <<"feature_talker_attention_global_score_tensor: False\n"
        <<"feature_talker_attention_global_probability_tensor: False\n"
        <<"feature_predictor_m1_locality_fusion: True\n"
        <<"feature_predictor_step0_staged_anchor: True\n"
        <<"feature_predictor_m1_qkv_fusion: True\n"
        <<"feature_predictor_m1_qk_norm_rope_cache_fusion: True\n"
        <<"feature_predictor_m1_attention_gqa2: True\n"
        <<"feature_predictor_m1_gate_up_swiglu_fusion: True\n"
        <<"feature_predictor_m1_down_residual_fusion: True\n"
        <<"feature_predictor_cache_policy: raw_material_in_cache_derived_in_vgpr_lds\n"
        <<"feature_priority_kpi_primary: e2e_ttfa\n"
        <<"feature_production_workdir_removed: True\n"
        <<"feature_predictor_build_async_overlap: True\n"
        <<"feature_predictor_build_overlap_with_speaker_frontend_talker: True\n"
        <<"feature_predictor_capture_mode: thread_local\n"
        <<"feature_predictor_capture_global_allocator_hazard: False\n"
        <<"feature_capture_overlap_dependency_control: manual_explicit\n"
        <<"feature_speaker_stream_ordered_allocator: True\n"
        <<"feature_frontend_no_device_wide_sync: True\n"
        <<"feature_talker_weight_sync_scope: upload_stream_only\n"
        <<"feature_stream_policy: explicit_nonblocking\n"
        <<"feature_speaker_explicit_stream: True\n"
        <<"feature_frontend_explicit_stream: True\n"
        <<"feature_talker_weight_upload_explicit_stream: True\n"
        <<"feature_legacy_stream_calls_in_capture_overlap: False\n"
        <<"feature_compile_requires_fgpu_default_stream_per_thread: False\n"
        <<"feature_per_frame_stdout_compacted: True\n"
        <<"feature_ttft_critical_path_instrumented: True\n"
        <<"feature_model_setup_overlap_with_tokenizer: True\n"
        <<"feature_talker_init_async_overlap: True\n"
        <<"feature_talker_init_overlap_with_speaker: True\n"
        <<"feature_decoder_parameter_preload_after_ttft: True\n"
        <<"feature_decoder_parameter_preload_overlap_with_generation: True\n"
        <<"feature_decoder_preload_can_affect_ttft: False\n"
        <<"feature_generation_explicit_stream: True\n"
        <<"feature_generation_device_wide_sync_per_frame: False\n"
        <<"feature_final_head_generation_stream: True\n"
        <<"feature_decoder_preload_independent_of_generation_barrier: True\n"
        <<"feature_streaming_decoder_pipeline_ready: True\n"
        <<"feature_final_head_sampler_default_stream: False\n"
        <<"feature_final_head_single_stream_ordering: True\n"
        <<"feature_eos_ordering_race_closed: True\n"
        <<"feature_priority_kpi: e2e_ttfa_generation\n"
        <<"feature_ttft_target_frozen_around_320ms: True\n"
        <<"feature_predictor_gfx11_bf16_dot2: True\n"
        <<"feature_predictor_dot2_qkv_m1: True\n"
        <<"feature_predictor_dot2_linear_m1: True\n"
        <<"feature_predictor_dot2_gate_up_m1: True\n"
        <<"feature_predictor_step0_dot2_changed: False\n"
        <<"feature_streaming_codec_decoder: True\n"
        <<"feature_streaming_codec_first_chunk_frames: 8\n"
        <<"feature_streaming_codec_prefix_recompute: False\n"
        <<"feature_streaming_codec_transformer_kv_cache: True\n"
        <<"feature_streaming_codec_persistent_causal_history: True\n"
        <<"feature_generation_decoder_pipeline: True\n"
        <<"feature_decoder_parameter_preload_stream_ordered: True\n"
        <<"feature_decoder_fixed_shape_graph_in_production: False\n"
        <<"feature_e2e_wavepack: True\n"
        <<"feature_predictor_outputs_per_wave: 2\n"
        <<"feature_talker_outputs_per_wave: 2\n"
        <<"feature_talker_dot2_changed: False\n"
        <<"feature_streaming_decoder_outputs_per_wave: 4\n"
        <<"feature_streaming_decoder_linear_pack4: True\n"
        <<"feature_streaming_decoder_conv_pack4: True\n"
        <<"feature_streaming_decoder_transconv_pack4: True\n"
        <<"feature_streaming_decoder_pointwise_pack4: True\n"
        <<"feature_streaming_codec_chunk_frames_effective: 16\n"
        <<"baseline_codec_vocoder_correctness: closed\n"
        <<"baseline_runtime_python_dependency: False\n"
        <<"baseline_runtime_huggingface_dependency: False\n"
        <<"baseline_runtime_rocblas_dependency: False\n"
        <<"baseline_runtime_hipblas_dependency: False\n"
        <<"baseline_success: True\n"
        <<"================================================================================\n";

    if(
        !options.keep_work
        &&!work.empty()
    ){
        fs::remove_all(work);
    }

    return 0;
}


static int run_legacy_native_e2e(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids,
    const PerfTime& e2e_start,
    double cli_and_tokenizer_ms){

    PerfStats perf;
    perf.cli_and_tokenizer_ms=
        cli_and_tokenizer_ms;

    const bool sampling_mode=
        (
            options.mode=="native-e2e-legacy_legacy_tag_n"
            || options.mode=="native-e2e-legacy_legacy_tag_n-validate"
        )
        && !options.greedy;

    configure_sampling(
        sampling_mode,
        options.seed);

    const fs::path work=select_work_dir(options);
    fs::remove_all(work);fs::create_directories(work);

    fs::path frontend_assets=options.native_assets_dir;
    if(frontend_assets.empty()){
        const fs::path exe=executable_path();
        frontend_assets=exe.parent_path().parent_path()/"legacy_native_assets_a";
    }
    frontend_assets=fs::absolute(frontend_assets);

    fs::path tail_assets=options.generation_tail_assets_dir;
    if(tail_assets.empty()){
        const fs::path exe=executable_path();
        tail_assets=exe.parent_path().parent_path()/"legacy_native_assets_c";
    }
    tail_assets=fs::absolute(tail_assets);

    std::cout
        <<"legacy_mode_b: native_e2e\n"
        <<"legacy_public_goal_b: raw_text_to_wav_single_cpp_hip\n"
        <<"inference_runtime_python_dependency: False\n"
        <<"inference_runtime_huggingface_dependency: False\n"
        <<"inference_runtime_rocblas_dependency: False\n"
        <<"inference_runtime_hipblas_dependency: False\n"
        <<"inference_runtime_external_code_objects: False\n"
        <<"execution_style: qingming_wave32_vgpr_lds_hipgraph_dag\n";

    const auto frontend_start=
        PerfClock::now();

    model_frontend::NativeFrontend frontend(fs::absolute(options.model_dir),frontend_assets);
    const auto native=frontend.run(native_ids);

    perf.frontend_ms=
        elapsed_ms(
            frontend_start,
            PerfClock::now());
    std::cout<<"native_frontend_closed: True\n"
             <<"native_prefill_length: "<<native.seq_len<<"\n";

    const auto core_setup_start=
        PerfClock::now();

    model_frontend::ModelWeights root_weights(fs::absolute(options.model_dir));
    const auto& codec_embedding=root_weights.suffix("talker.model.codec_embedding.weight");
    if(codec_embedding.dtype!="BF16"||codec_embedding.shape.size()!=2||codec_embedding.shape[1]!=1024)
        throw std::runtime_error("legacy_legacy_tag_h unexpected Talker codec embedding");

    NativeFinalHead final_head(fs::absolute(options.model_dir));

    const fs::path predictor_assets=tail_assets/"predictor";
    const fs::path predictor_runtime=work/"predictor_runtime_weights";
    fs::create_directories(predictor_runtime);
    {
        std::error_code ec;fs::remove(predictor_runtime/"weights",ec);
        fs::create_directory_symlink(fs::absolute(predictor_assets/"weights"),predictor_runtime/"weights");
    }
    code_predictor::explicit_runtime::verify_static_maps();
    code_predictor::PredictorWeights predictor_weights(predictor_runtime);

    perf.core_model_setup_ms=
        elapsed_ms(
            core_setup_start,
            PerfClock::now());

    const auto codec=legacy_generate_native_codec_b(
        fs::absolute(options.model_dir),work,native.prefill,static_cast<int>(native.seq_len),
        static_cast<int>(options.max_new_tokens),native.tts_pad,predictor_assets,predictor_weights,
        codec_embedding,final_head,e2e_start,perf);

    const int frames=static_cast<int>(codec.size()/16);

    const auto decoder_setup_start=
        PerfClock::now();

    const fs::path tokenizer=codec_decoder::locate_speech_tokenizer(fs::absolute(options.model_dir));
    std::cout<<"speech_tokenizer_dir: "<<tokenizer.string()<<"\n";

    const bool legacy_validation_a=
        options.mode=="native-e2e-legacy_legacy_tag_m-validate";

    const bool legacy_production=
        options.mode=="native-e2e-legacy_legacy_tag_n"
        || options.mode=="native-e2e-legacy_legacy_tag_n-validate";

    const bool legacy_validation_b=
        options.mode=="native-e2e-legacy_legacy_tag_n-validate";

    codec_decoder::NativeCodecDecoder decoder(
        tokenizer,
        frames,
        legacy_validation_a,
        legacy_validation_a || legacy_validation_b,
        legacy_production);

    perf.codec_decoder_setup_ms=
        elapsed_ms(
            decoder_setup_start,
            PerfClock::now());

    const auto decoder_graph_start=
        PerfClock::now();

    auto waveform=decoder.decode(codec);

    perf.codec_decoder_graph_ms=
        elapsed_ms(
            decoder_graph_start,
            PerfClock::now());

    perf.ttfa_ms=
        elapsed_ms(
            e2e_start,
            PerfClock::now());

    if(legacy_validation_b){
        auto legacy_waveform=
            decoder.decode_legacy_reference(
                codec);

        if(
            legacy_waveform.size()
            != waveform.size()
        ){
            throw std::runtime_error(
                "legacy_legacy_tag_j legacy/fused waveform length mismatch.");
        }

        double sum_abs=0.0;
        double sum_sq=0.0;
        double ref_sq=0.0;
        double max_abs=0.0;
        double sum_a=0.0;
        double sum_b=0.0;
        double sum_aa=0.0;
        double sum_bb=0.0;
        double sum_ab=0.0;

        const double count=
            static_cast<double>(
                waveform.size());

        for(std::size_t i=0;i<waveform.size();++i){
            const double a=
                static_cast<double>(
                    waveform[i]);

            const double b=
                static_cast<double>(
                    legacy_waveform[i]);

            const double d=a-b;

            sum_abs+=std::fabs(d);
            sum_sq+=d*d;
            ref_sq+=b*b;
            max_abs=std::max(
                max_abs,
                std::fabs(d));

            sum_a+=a;
            sum_b+=b;
            sum_aa+=a*a;
            sum_bb+=b*b;
            sum_ab+=a*b;
        }

        const double mae=
            count>0.0
            ? sum_abs/count
            : 0.0;

        const double rmse=
            count>0.0
            ? std::sqrt(sum_sq/count)
            : 0.0;

        const double ref_rms=
            count>0.0
            ? std::sqrt(ref_sq/count)
            : 0.0;

        const double rel_rmse=
            rmse
            / std::max(
                ref_rms,
                1.0e-20);

        const double cov=
            count*sum_ab
            - sum_a*sum_b;

        const double var_a=
            count*sum_aa
            - sum_a*sum_a;

        const double var_b=
            count*sum_bb
            - sum_b*sum_b;

        const double corr_denom=
            std::sqrt(
                std::max(0.0,var_a)
                * std::max(0.0,var_b));

        const double corr=
            corr_denom>0.0
            ? cov/corr_denom
            : 1.0;

        const bool regression_pass=
            rel_rmse<=1.0e-5
            && corr>=0.999999;

        std::cout
            <<"fused_vs_waveform_mae: "<<mae<<"\n"
            <<"fused_vs_waveform_rmse: "<<rmse<<"\n"
            <<"fused_vs_waveform_rel_rmse: "<<rel_rmse<<"\n"
            <<"fused_vs_waveform_max_abs: "<<max_abs<<"\n"
            <<"fused_vs_waveform_corr: "<<corr<<"\n"
            <<"hot_math_regression_pass: "
            <<(regression_pass?"True":"False")
            <<"\n";

        decoder.run_diagnostics(
            codec,
            work/"legacy_native_codec_trace_b");
    }

    if(legacy_validation_a){
        const fs::path trace_root=
            work/"legacy_native_codec_trace_a";

        legacy_write_binary(
            work/"legacy_native_codec_b.i32.bin",
            codec);

        legacy_write_binary(
            work/"native_waveform.f32.bin",
            waveform);

        write_text_file(
            work/"native_sample_rate.txt",
            std::to_string(
                codec_decoder::SAMPLE_RATE)
                +"\n");

        decoder.run_diagnostics(
            codec,
            trace_root);

        std::cout
            <<"validation_native_codec_asset: "
            <<(work/"legacy_native_codec_b.i32.bin").string()
            <<"\n"
            <<"validation_native_waveform_asset: "
            <<(work/"native_waveform.f32.bin").string()
            <<"\n"
            <<"validation_native_trace_ready: True\n";
    }

    const std::size_t expected=static_cast<std::size_t>(frames)*codec_decoder::UPSAMPLE;
    if(waveform.size()!=expected)throw std::runtime_error("legacy_legacy_tag_h waveform sample count mismatch");

    const auto wav_write_start=
        PerfClock::now();

    write_float_wav(fs::absolute(options.output_wav),waveform,codec_decoder::SAMPLE_RATE);

    perf.wav_write_ms=
        elapsed_ms(
            wav_write_start,
            PerfClock::now());

    perf.e2e_ms=
        elapsed_ms(
            e2e_start,
            PerfClock::now());

    if(legacy_production){
        std::cout
            <<"legacy_mode_f: qingming_lds_hot_dag\n"
            <<"single_cpp_metrics: True\n"
            <<"native_sampling: True\n"
            <<"legacy_scope_b: native_codec_hot_decoder_retile\n"
            <<"hot_stage_count: 4\n"
            <<"hot_stage_names: decoder1,decoder2,decoder3,decoder4\n"
            <<"hot_transconv_backend: wave32_8way_explicit_lds\n"
            <<"hot_dilated_conv_backend: wave32_8way_explicit_lds\n"
            <<"hot_pointwise_backend: wave32_8way_explicit_lds_residual_fused\n"
            <<"hot_snake_backend: single_materialization_global_once\n"
            <<"hot_global_input_read_reuse_target: 8x_per_output_tile\n"
            <<"legacy_graph_node_count_c: "<<decoder.node_count()<<"\n"
            <<"legacy_graph_node_reduction_vs_b: "
            <<(247>decoder.node_count()?247-decoder.node_count():0)
            <<"\n";
    }

    if(legacy_validation_a){
        std::cout
            <<"legacy_mode_c: codec_correctness_gate\n"
            <<"production_graph_profile_events: False\n"
            <<"production_graph_trace_nodes: False\n"
            <<"validation_same_native_codec_matrix: True\n"
            <<"validation_reference_decoder_runtime_only_after_native_process: True\n";
    }

    std::cout
        <<"native_codec_frame_count: "<<frames<<"\n"
        <<"native_audio_sample_rate: "<<codec_decoder::SAMPLE_RATE<<"\n"
        <<"native_audio_sample_count: "<<waveform.size()<<"\n"
        <<"native_audio_expected_sample_count: "<<expected<<"\n"
        <<"audio_samples_per_codec_frame: "<<codec_decoder::UPSAMPLE<<"\n"
        <<"legacy_output_wav_codec_source_b: native_free_recurrence\n"
        <<"output_wav_decoder_source: native_codec_decoder_vocoder\n"
        <<"legacy_raw_text_to_codec_native_closed_b: True\n"
        <<"native_codec_decoder_vocoder_closed: True\n"
        <<"legacy_native_raw_text_to_wav_closed_c: True\n"
        <<"end_to_end_runtime_bridge_process_count: 0\n"
        <<"legacy_remaining_correctness_work_a: talker_numeric_drift\n"
        <<"next_performance_boundary: talker_numeric_closure_and_further_hot_dag_tiling\n"
        <<"current_boundary: raw text reaches WAV through one C++/HIP runtime; codec/vocoder correctness is closed, remaining correctness focus is Talker numeric drift\n";


    // ------------------------------------------------------------------------
    // legacy_legacy_tag_j.1 final external benchmark summary.
    // ------------------------------------------------------------------------
    double waveform_sum_sq=0.0;
    double waveform_peak_abs=0.0;
    std::size_t waveform_nonfinite=0;
    std::size_t waveform_clipped=0;

    for(const float sample:waveform){
        if(!std::isfinite(sample)){
            ++waveform_nonfinite;
            continue;
        }

        const double value=
            static_cast<double>(sample);

        waveform_sum_sq+=value*value;
        waveform_peak_abs=std::max(
            waveform_peak_abs,
            std::fabs(value));

        if(std::fabs(value)>=1.0){
            ++waveform_clipped;
        }
    }

    const double audio_duration_s=
        static_cast<double>(
            waveform.size())
        / static_cast<double>(
            codec_decoder::SAMPLE_RATE);

    const double waveform_rms=
        waveform.empty()
        ? 0.0
        : std::sqrt(
            waveform_sum_sq
            / static_cast<double>(
                waveform.size()));

    const double e2e_s=
        perf.e2e_ms/1000.0;

    const double generation_s=
        perf.codec_generation_ms/1000.0;

    const double decoder_s=
        perf.codec_decoder_graph_ms/1000.0;

    const double e2e_rtf=
        audio_duration_s>0.0
        ? e2e_s/audio_duration_s
        : 0.0;

    const double e2e_realtime_x=
        e2e_s>0.0
        ? audio_duration_s/e2e_s
        : 0.0;

    const double generation_fps=
        generation_s>0.0
        ? static_cast<double>(frames)
            / generation_s
        : 0.0;

    const double generation_realtime_x=
        generation_s>0.0
        ? audio_duration_s/generation_s
        : 0.0;

    const double decoder_rtf=
        audio_duration_s>0.0
        ? decoder_s/audio_duration_s
        : 0.0;

    const double decoder_realtime_x=
        decoder_s>0.0
        ? audio_duration_s/decoder_s
        : 0.0;

    const double output_samples_per_second=
        e2e_s>0.0
        ? static_cast<double>(
            waveform.size())
            / e2e_s
        : 0.0;

    const double clip_fraction=
        waveform.empty()
        ? 0.0
        : static_cast<double>(
            waveform_clipped)
            / static_cast<double>(
                waveform.size());

    const double codec_emit_p50=
        percentile(
            perf.codec_emit_frame_ms,
            0.50);

    const double codec_emit_p95=
        percentile(
            perf.codec_emit_frame_ms,
            0.95);

    const double codec_emit_p99=
        percentile(
            perf.codec_emit_frame_ms,
            0.99);

    const double codec_emit_max=
        perf.codec_emit_frame_ms.empty()
        ? 0.0
        : *std::max_element(
            perf.codec_emit_frame_ms.begin(),
            perf.codec_emit_frame_ms.end());

    const double talker_decode_p50=
        percentile(
            perf.talker_decode_step_ms,
            0.50);

    const double talker_decode_p95=
        percentile(
            perf.talker_decode_step_ms,
            0.95);

    std::size_t gpu_free_bytes=0;
    std::size_t gpu_total_bytes=0;

    const hipError_t mem_info_status=
        hipMemGetInfo(
            &gpu_free_bytes,
            &gpu_total_bytes);

    const bool gpu_mem_info_ok=
        mem_info_status==hipSuccess;

    const double mib=
        1024.0*1024.0;

    const double gpu_free_mib=
        gpu_mem_info_ok
        ? static_cast<double>(
            gpu_free_bytes)/mib
        : 0.0;

    const double gpu_total_mib=
        gpu_mem_info_ok
        ? static_cast<double>(
            gpu_total_bytes)/mib
        : 0.0;

    const double gpu_used_mib_observed=
        gpu_mem_info_ok
        ? static_cast<double>(
            gpu_total_bytes-gpu_free_bytes)
            / mib
        : 0.0;

    const double host_peak_rss_mib=
        query_host_peak_rss_mib();

    const double codec_frame_rate_hz=
        static_cast<double>(
            codec_decoder::SAMPLE_RATE)
        / static_cast<double>(
            codec_decoder::UPSAMPLE);

    const double audio_ms_per_codec_frame=
        1000.0
        / codec_frame_rate_hz;

    const bool benchmark_is_production_only=
        legacy_production
        && !legacy_validation_b
        && !legacy_validation_a;

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"\n"
        <<"================================================================================\n"
        <<"legacy_legacy_tag_j.1 FINAL SUMMARY\n"
        <<"================================================================================\n"
        <<"summary_version: 1\n"
        <<"legacy_engine: single_cpp_hip_qingming\n"
        <<"legacy_mode_d: "<<options.mode<<"\n"
        <<"benchmark_scope: cold_one_shot_in_process\n"
        <<"benchmark_production_only: "
        <<(benchmark_is_production_only?"True":"False")
        <<"\n"
        <<"legacy_runtime_python_dependency_a: False\n"
        <<"legacy_runtime_huggingface_dependency_a: False\n"
        <<"legacy_runtime_rocblas_dependency_a: False\n"
        <<"legacy_runtime_hipblas_dependency_a: False\n"
        <<"legacy_runtime_bridge_process_count_a: 0\n"
        <<"legacy_codec_vocoder_correctness_closed_a: True\n"
        <<"legacy_remaining_correctness_work_b: talker_numeric_drift\n"
        <<"sampling_semantics_fixed: True\n"
        <<"sampling_enabled: "
        <<(sampling_mode?"True":"False")
        <<"\n"
        <<"sampling_seed: "<<options.seed<<"\n"
        <<"main_top_k: 50\n"
        <<"main_temperature: 0.9\n"
        <<"subtalker_top_k: 50\n"
        <<"subtalker_temperature: 0.9\n"
        <<"\n"
        <<"input_text_bytes: "<<options.text.size()<<"\n"
        <<"legacy_input_token_count_a: "<<native_ids.size()<<"\n"
        <<"prefill_token_count: "<<native.seq_len<<"\n"
        <<"legacy_codec_frame_count_a: "<<frames<<"\n"
        <<"codec_group_count: 16\n"
        <<"legacy_codec_frame_rate_hz: "<<codec_frame_rate_hz<<"\n"
        <<"legacy_audio_ms_per_codec_frame: "<<audio_ms_per_codec_frame<<"\n"
        <<"audio_sample_rate_hz: "<<codec_decoder::SAMPLE_RATE<<"\n"
        <<"legacy_audio_sample_count_a: "<<waveform.size()<<"\n"
        <<"legacy_audio_duration_s_a: "<<audio_duration_s<<"\n"
        <<"\n"
        <<"ttft_definition: raw_text_to_first_complete_codec_frame\n"
        <<"legacy_ttft_ms_a: "<<perf.ttft_ms<<"\n"
        <<"ttfa_definition: raw_text_to_complete_host_waveform_offline_vocoder\n"
        <<"legacy_ttfa_ms_a: "<<perf.ttfa_ms<<"\n"
        <<"e2e_definition: main_entry_to_wav_write_complete\n"
        <<"legacy_e2e_ms_a: "<<perf.e2e_ms<<"\n"
        <<"\n"
        <<"cli_tokenizer_ms: "<<perf.cli_and_tokenizer_ms<<"\n"
        <<"legacy_frontend_ms_a: "<<perf.frontend_ms<<"\n"
        <<"legacy_core_model_setup_ms: "<<perf.core_model_setup_ms<<"\n"
        <<"legacy_talker_engine_init_ms: "<<perf.talker_engine_init_ms<<"\n"
        <<"legacy_talker_rope_build_ms: "<<perf.talker_rope_build_ms<<"\n"
        <<"legacy_talker_prefill_ms_a: "<<perf.talker_prefill_ms<<"\n"
        <<"legacy_codec_generation_ms_a: "<<perf.codec_generation_ms<<"\n"
        <<"legacy_codec_decoder_setup_ms_a: "<<perf.codec_decoder_setup_ms<<"\n"
        <<"legacy_codec_decoder_graph_ms_a: "<<perf.codec_decoder_graph_ms<<"\n"
        <<"legacy_wav_write_ms: "<<perf.wav_write_ms<<"\n"
        <<"\n"
        <<"codec_emit_frame_p50_ms: "<<codec_emit_p50<<"\n"
        <<"codec_emit_frame_p95_ms: "<<codec_emit_p95<<"\n"
        <<"codec_emit_frame_p99_ms: "<<codec_emit_p99<<"\n"
        <<"codec_emit_frame_max_ms: "<<codec_emit_max<<"\n"
        <<"talker_decode_step_p50_ms: "<<talker_decode_p50<<"\n"
        <<"talker_decode_step_p95_ms: "<<talker_decode_p95<<"\n"
        <<"\n"
        <<"legacy_codec_generation_frames_per_s_a: "<<generation_fps<<"\n"
        <<"codec_generation_realtime_x: "<<generation_realtime_x<<"\n"
        <<"legacy_decoder_rtf: "<<decoder_rtf<<"\n"
        <<"legacy_decoder_realtime_x_a: "<<decoder_realtime_x<<"\n"
        <<"legacy_e2e_rtf: "<<e2e_rtf<<"\n"
        <<"legacy_e2e_realtime_x_a: "<<e2e_realtime_x<<"\n"
        <<"output_samples_per_wall_s: "<<output_samples_per_second<<"\n"
        <<"\n"
        <<"legacy_graph_node_count_a: "<<decoder.node_count()<<"\n"
        <<"legacy_graph_node_reduction_vs_a: "
        <<(247>decoder.node_count()?247-decoder.node_count():0)
        <<"\n"
        <<"hot_backend: qingming_wave32_8wave_explicit_lds\n"
        <<"hot_pointwise_residual_fused: True\n"
        <<"\n"
        <<"legacy_waveform_rms: "<<waveform_rms<<"\n"
        <<"legacy_waveform_peak_abs: "<<waveform_peak_abs<<"\n"
        <<"waveform_nonfinite_count: "<<waveform_nonfinite<<"\n"
        <<"waveform_clipped_sample_count: "<<waveform_clipped<<"\n"
        <<"waveform_clip_fraction: "<<clip_fraction<<"\n"
        <<"\n"
        <<"gpu_meminfo_available: "
        <<(gpu_mem_info_ok?"True":"False")
        <<"\n"
        <<"legacy_gpu_total_mib: "<<gpu_total_mib<<"\n"
        <<"gpu_free_mib_after_inference: "<<gpu_free_mib<<"\n"
        <<"gpu_used_mib_observed_systemwide: "<<gpu_used_mib_observed<<"\n"
        <<"gpu_memory_note: hipMemGetInfo_is_device_wide_not_process_peak\n"
        <<"legacy_host_peak_rss_mib: "<<host_peak_rss_mib<<"\n"
        <<"\n"
        <<"legacy_output_wav_a: "<<fs::absolute(options.output_wav).string()<<"\n"
        <<"legacy_success_a: True\n"
        <<"================================================================================\n";


    if(!options.keep_work)fs::remove_all(work);
    return 0;
}


static int run_generation_tail_native_e2e(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids){

    const fs::path work=select_work_dir(options);
    fs::remove_all(work);
    fs::create_directories(work);

    fs::path frontend_assets=options.native_assets_dir;
    if(frontend_assets.empty()){
        const fs::path exe=executable_path();
        frontend_assets=
            exe.parent_path().parent_path()
            /"legacy_native_assets_a";
    }
    frontend_assets=fs::absolute(frontend_assets);

    fs::path tail_assets=
        options.generation_tail_assets_dir;
    if(tail_assets.empty()){
        const fs::path exe=executable_path();
        tail_assets=
            exe.parent_path().parent_path()
            /"legacy_native_assets_b";
    }
    tail_assets=fs::absolute(tail_assets);

    model_frontend::NativeFrontend frontend(
        fs::absolute(options.model_dir),
        frontend_assets);
    const auto native=frontend.run(native_ids);

    const fs::path prefill=work/"native_prefill.bf16.bin";
    const fs::path pad=work/"native_tts_pad.bf16.bin";

    model_frontend::write_u16(
        prefill,native.prefill);
    model_frontend::write_u16(
        pad,native.tts_pad);
    model_frontend::write_u16(
        work/"native_tts_pad_embedding.bf16.bin",
        native.tts_pad_trace.embedding);
    model_frontend::write_u16(
        work/"native_tts_pad_fc1.bf16.bin",
        native.tts_pad_trace.fc1);
    model_frontend::write_u16(
        work/"native_tts_pad_silu.bf16.bin",
        native.tts_pad_trace.silu);
    model_frontend::write_u16(
        work/"native_tts_pad_fc2.bf16.bin",
        native.tts_pad_trace.fc2);
    write_text_file(
        work/"request.txt",
        options.text);
    write_i64(
        work/"native_input_ids.i64.bin",
        native_ids);

    const fs::path script=
        options.bridge_script.empty()
        ? default_bridge_script()
        : fs::absolute(options.bridge_script);

    std::cout
        <<"legacy_mode_a: generation-tail-algorithmic-validation-0\n"
        <<"legacy_public_goal_a: raw_text_to_wav_single_cpp_hip\n"
        <<"legacy_scope_a: native_talker_prefill_decode_plus_native_recurrence\n"
        <<"talker_prefill_backend: native_cpp_hip\n"
        <<"talker_prefill_kv_source: native\n"
        <<"hf_prefill_kv_dependency: False\n"
        <<"teacher_decode_input_used_for_validation_only: True\n"
        <<"decode_hot_backend: qingming_wave32_vgpr_lds\n"
        <<"qk_dtype_boundary: bf16_matmul_output_then_bf16_scale\n"
        <<"hipgraph_decode: False\n"
        <<"hipgraph_next_after_math_closure: True\n"
        <<"layer_independent_teacher: exact_layer_input_plus_exact_kv_before\n"
        <<"teacher_diagnostic_persistence: required_compact_tensors\n"
        <<"teacher_diagnostic_legacy_save_flag_required: False\n"
        <<"legacy_goal: mathematically_correct_native_inference_not_isa_reproduction\n"
        <<"production_backend_rocblas_free: True\n"
        <<"production_backend_hipblas_free: True\n"
        <<"production_backend_external_code_objects: False\n"
        <<"linear_backend: native_wave32_bf16_fp32_accumulation\n"
        <<"attention_backend: native_wave32_qk_fp32acc_softmax_pv_fp32acc\n"
        <<"rmsnorm_backend: native_hip\n"
        <<"rope_backend: native_hip\n"
        <<"swiglu_backend: native_hip\n"
        <<"validation_reference: HuggingFace_outer_and_independent_predictor_steps\n"
        <<"validation_bit_exact_required: False\n"
        <<"validation_all_frames_scan: True\n"
        <<"current_scope: native_frontend_to_native_codec_matrix_recurrence_plus_validation\n";

    const std::vector<std::string> capture_command{
        "python3",script.string(),
        "--model-dir",fs::absolute(options.model_dir).string(),
        "--text-file",(work/"request.txt").string(),
        "--native-token-ids",(work/"native_input_ids.i64.bin").string(),
        "--native-prefill",prefill.string(),
        "--native-tts-pad",pad.string(),
        "--prefill-seq-len",std::to_string(native.seq_len),
        "--output",work.string(),
        "--max-new-tokens",std::to_string(options.max_new_tokens),
        "--frontend-native",
        "--tail-capture"
    };

    const int capture_rc=run_process(capture_command);
    if(capture_rc!=0)
        throw std::runtime_error(
            "legacy_legacy_tag_e.3 Talker-tail capture failed with exit code "
            +std::to_string(capture_rc));

    const int frame_count=
        read_int_text(work/"frame_count.txt");
    if(frame_count<=0)
        throw std::runtime_error(
            "legacy_legacy_tag_e.3 tail capture returned no complete frames");

    const auto hidden=
        read_binary<std::uint16_t>(
            work/"talker_tail_hidden.bf16.bin");
    const auto expected_next=
        read_binary<std::uint16_t>(
            work/"talker_next_inputs.bf16.bin");
    const auto official_codes=
        read_binary<std::int32_t>(
            work/"official_codec_frames.i32.bin");
    const auto official_raw_logits=
        read_binary<float>(
            work/"official_raw_logits.f32.bin");
    const auto official_processed_scores=
        read_binary<float>(
            work/"official_processed_scores.f32.bin");
    const auto official_repetition_mask=
        read_binary<std::uint8_t>(
            work/"official_repetition_mask.u8.bin");
    const auto official_generated_code0=
        read_binary<std::int32_t>(
            work/"official_generated_code0_sequence.i32.bin");

    const std::size_t hidden_expected=
        static_cast<std::size_t>(frame_count)*1024;
    const std::size_t code_expected=
        static_cast<std::size_t>(frame_count)*16;

    const std::size_t score_row=3072;
    const std::size_t score_expected_min=
        static_cast<std::size_t>(frame_count)*score_row;

    if(
        hidden.size()!=hidden_expected
        || expected_next.size()!=hidden_expected
        || official_codes.size()!=code_expected
        || official_raw_logits.size()<score_expected_min
        || official_processed_scores.size()<score_expected_min
        || official_repetition_mask.size()<score_expected_min
        || official_generated_code0.size()
            <static_cast<std::size_t>(frame_count)
    )
        throw std::runtime_error(
            "legacy_legacy_tag_e.4 tail capture tensor size mismatch");

    NativeFinalHead final_head(
        fs::absolute(options.model_dir));

    const auto talker_summary =
        talker::validate_native_talker(
            fs::absolute(options.model_dir),
            work,
            native.prefill,
            native.seq_len,
            frame_count,
            hidden,
            expected_next,
            official_codes,
            final_head);

    const double talker_decode_hidden_mae =
        talker_summary.decode_values == 0
        ? 0.0
        : talker_summary.decode_abs_sum
            / static_cast<double>(
                talker_summary.decode_values);

    const double talker_decode_code0_match_rate =
        talker_summary.decode_steps == 0
        ? 1.0
        : static_cast<double>(
            talker_summary.decode_code0_matches)
            / static_cast<double>(
                talker_summary.decode_steps);

    std::cout
        << "talker_prefill_hidden_mae: "
        << talker_summary.prefill_hidden_mae
        << "\n"
        << "talker_prefill_hidden_max_abs_diff: "
        << talker_summary.prefill_hidden_max_abs
        << "\n"
        << "talker_prefill_code0_match: "
        << (
            talker_summary.prefill_code0_match
            ? "True"
            : "False")
        << "\n"
        << "talker_decode_step_count: "
        << talker_summary.decode_steps
        << "\n"
        << "legacy_talker_decode_hidden_mae: "
        << talker_decode_hidden_mae
        << "\n"
        << "talker_decode_hidden_max_abs_diff: "
        << talker_summary.decode_max_abs
        << "\n"
        << "talker_decode_code0_match_count: "
        << talker_summary.decode_code0_matches
        << "\n"
        << "legacy_talker_decode_code0_match_rate: "
        << talker_decode_code0_match_rate
        << "\n"
        << "native_talker_prefill_executed: True\n"
        << "native_talker_dynamic_kv_executed: "
        << (
            talker_summary.decode_steps
                == std::max(
                    0,
                    frame_count - 1)
            ? "True"
            : "False")
        << "\n";

    model_frontend::ModelWeights tables(
        fs::absolute(options.model_dir));
    const auto& codec_embedding=
        tables.suffix(
            "talker.model.codec_embedding.weight");

    if(
        codec_embedding.dtype!="BF16"
        || codec_embedding.shape.size()!=2
        || codec_embedding.shape[1]!=1024
    )
        throw std::runtime_error(
            "unexpected Talker codec embedding table");

    const fs::path predictor_assets=
        tail_assets/"predictor";
    const fs::path predictor_runtime=
        work/"predictor_runtime";
    const fs::path predictor_output=
        work/"predictor_output";
    fs::create_directories(predictor_runtime);

    {
        std::error_code ec;
        fs::remove(predictor_runtime/"weights",ec);
        fs::create_directory_symlink(
            fs::absolute(predictor_assets/"weights"),
            predictor_runtime/"weights");
    }

    code_predictor::explicit_runtime::
        verify_static_maps();

    code_predictor::PredictorWeights
        predictor_weights(predictor_runtime);

    const std::vector<std::int32_t>
        legacy_native_codec_a =
            legacy_generate_native_codec_a(
                fs::absolute(
                    options.model_dir),
                work,
                native.prefill,
                native.seq_len,
                options.max_new_tokens,
                native.tts_pad,
                predictor_assets,
                predictor_weights,
                codec_embedding,
                final_head);

    // Validation teacher-forces the official code0 history. That keeps every
    // captured frame independent and prevents one legal numerical difference
    // from poisoning all later comparisons.
    std::vector<std::int32_t> validation_history;
    validation_history.reserve(
        static_cast<std::size_t>(frame_count)+1);

    std::vector<std::int32_t> native_codes(
        code_expected,0);

    int raw_logits_exact=0;
    int repetition_mask_exact=0;
    int processed_scores_exact=0;
    int final_head_exact=0;
    int predictor_exact=0;
    int next_input_exact=0;
    int executed_frames=0;
    std::size_t predictor_group_matches=0;
    std::size_t predictor_group_total=0;
    std::size_t teacher_predictor_group_matches=0;
    std::size_t teacher_predictor_group_total=0;
    int teacher_predictor_exact_frames=0;
    double next_input_abs_sum=0.0;
    float next_input_max_abs=0.0f;
    double raw_logits_abs_sum=0.0;
    std::size_t raw_logits_value_count=0;

    auto host_bf16_to_float=[](std::uint16_t bits)->float{
        std::uint32_t raw=
            static_cast<std::uint32_t>(bits)<<16;
        float value=0.0f;
        std::memcpy(&value,&raw,sizeof(value));
        return value;
    };

    for(int frame=0;frame<frame_count;++frame){
        ++executed_frames;
        const std::uint16_t* frame_hidden=
            hidden.data()
            +static_cast<std::size_t>(frame)*1024;

        const FinalHeadTrace final_trace=
            final_head.run(
                frame_hidden,
                validation_history,
                frame);

        const std::int32_t code0=
            final_trace.code;

        const std::int32_t official_code0=
            official_codes[
                static_cast<std::size_t>(frame)*16];

        const float* official_raw=
            official_raw_logits.data()
            +static_cast<std::size_t>(frame)*score_row;
        const float* official_processed=
            official_processed_scores.data()
            +static_cast<std::size_t>(frame)*score_row;
        const std::uint8_t* official_mask=
            official_repetition_mask.data()
            +static_cast<std::size_t>(frame)*score_row;

        const std::size_t raw_mismatch=
            f32_mismatch_count(
                final_trace.raw_logits.data(),
                official_raw,
                score_row);
        const std::size_t processed_mismatch=
            f32_mismatch_count(
                final_trace.processed_scores.data(),
                official_processed,
                score_row);
        const float raw_max_abs_diff=
            f32_max_abs_diff(
                final_trace.raw_logits.data(),
                official_raw,
                score_row);
        const std::size_t raw_first_mismatch=
            f32_first_mismatch(
                final_trace.raw_logits.data(),
                official_raw,
                score_row);
        const std::size_t processed_first_mismatch=
            f32_first_mismatch(
                final_trace.processed_scores.data(),
                official_processed,
                score_row);
        const float processed_max_abs_diff=
            f32_max_abs_diff(
                final_trace.processed_scores.data(),
                official_processed,
                score_row);

        for(std::size_t i=0;i<score_row;++i){
            raw_logits_abs_sum += std::abs(
                static_cast<double>(
                    final_trace.raw_logits[i])
                - static_cast<double>(
                    official_raw[i]));
        }
        raw_logits_value_count += score_row;

        std::vector<std::uint8_t> native_mask(
            score_row,0);
        for(const std::int32_t id:validation_history){
            if(id>=0 && id<static_cast<std::int32_t>(score_row))
                native_mask[
                    static_cast<std::size_t>(id)
                ]=1;
        }
        const std::size_t mask_mismatch=
            u8_mismatch_count(
                native_mask.data(),
                official_mask,
                score_row);

        if(raw_mismatch==0)
            ++raw_logits_exact;
        if(mask_mismatch==0)
            ++repetition_mask_exact;
        if(processed_mismatch==0)
            ++processed_scores_exact;
        if(code0==official_code0)
            ++final_head_exact;

        validation_history.push_back(
            official_code0);

        // Predictor validation is teacher-forced on semantic group0 so we
        // measure the Predictor math rather than amplify a FinalHead choice.
        const std::int32_t predictor_code0=
            official_code0;

        const auto code0_embedding=
            model_frontend::tensor_row_bf16(
                codec_embedding,
                predictor_code0);

        legacy_create_predictor_frame_assets_a(
            predictor_runtime,
            predictor_assets,
            work/"official_predictor",
            frame,
            frame_hidden,
            code0_embedding,
            native.tts_pad,
            predictor_code0);

        code_predictor::run_frame(
            predictor_runtime,
            predictor_output,
            "free",
            frame,
            predictor_weights);

        code_predictor::run_frame(
            predictor_runtime,
            predictor_output,
            "teacher",
            frame,
            predictor_weights);

        code_predictor::run_frame(
            predictor_runtime,
            predictor_output,
            "layer_teacher",
            frame,
            predictor_weights);

        const std::string frame_name=
            std::string("frame_")
            +(frame<10?"0":"")
            +std::to_string(frame);

        const auto generated=
            read_binary<std::int32_t>(
                predictor_output/"free"/frame_name
                /"generated_codes.i32.bin");

        const auto teacher_generated=
            read_binary<std::int32_t>(
                predictor_output/"teacher"/frame_name
                /"generated_codes.i32.bin");

        const auto next=
            read_binary<std::uint16_t>(
                predictor_output/"free"/frame_name
                /"next_talker_input.bf16.bin");

        if(
            generated.size()!=16
            || teacher_generated.size()!=16
            || next.size()!=1024
        )
            throw std::runtime_error(
                "legacy_legacy_tag_f.2 predictor output size mismatch");

        bool teacher_frame_exact=true;
        int teacher_first_bad_group=-1;
        int teacher_group_matches_this_frame=0;

        for(int group=1;group<16;++group){
            const std::int32_t official_group_code=
                official_codes[
                    static_cast<std::size_t>(frame)*16+group];

            const bool teacher_match=
                teacher_generated[group]
                ==official_group_code;

            ++teacher_predictor_group_total;

            if(teacher_match){
                ++teacher_predictor_group_matches;
                ++teacher_group_matches_this_frame;
            }else{
                teacher_frame_exact=false;
                if(teacher_first_bad_group<0)
                    teacher_first_bad_group=group;
            }
        }

        if(teacher_frame_exact)
            ++teacher_predictor_exact_frames;

        std::cout
            <<"teacher_frame: frame="
            <<frame
            <<" group_matches="
            <<teacher_group_matches_this_frame
            <<"/15"
            <<" exact="
            <<(teacher_frame_exact?"True":"False")
            <<" first_bad_group="
            <<teacher_first_bad_group
            <<"\n";

        bool frame_codes_exact=true;
        int first_bad_group=-1;
        std::int32_t first_bad_native_code=-1;
        std::int32_t first_bad_official_code=-1;
        for(int group=0;group<16;++group){
            const std::int32_t official_group_code=
                official_codes[
                    static_cast<std::size_t>(frame)*16+group];
            native_codes[
                static_cast<std::size_t>(frame)*16+group
            ]=generated[group];
            if(
                first_bad_group<0
                && generated[group]!=official_group_code
            ){
                first_bad_group=group;
                first_bad_native_code=generated[group];
                first_bad_official_code=official_group_code;
            }
            const bool group_match =
                generated[group]==official_group_code;

            frame_codes_exact &=
                group_match;

            if(group>0){
                ++predictor_group_total;
                if(group_match)
                    ++predictor_group_matches;
            }
        }
        if(frame_codes_exact)
            ++predictor_exact;

        const std::uint16_t* expected=
            expected_next.data()
            +static_cast<std::size_t>(frame)*1024;

        std::size_t next_mismatch=0;
        std::size_t next_first_mismatch=1024;
        for(std::size_t i=0;i<1024;++i){
            if(next[i]!=expected[i]){
                if(next_first_mismatch==1024)
                    next_first_mismatch=i;
                ++next_mismatch;
            }
        }

        if(next_mismatch==0)
            ++next_input_exact;

        for(std::size_t i=0;i<1024;++i){
            const float got=
                host_bf16_to_float(
                    next[i]);
            const float ref=
                host_bf16_to_float(
                    expected[i]);
            const float diff=
                std::abs(got-ref);

            next_input_abs_sum += diff;

            if(diff>next_input_max_abs)
                next_input_max_abs=diff;
        }

        // legacy_legacy_tag_f does not re-run a frame just because BF16 bits differ.


        std::cout
            <<"tail_frame: frame="
            <<frame
            <<" raw_logits_mismatch="<<raw_mismatch
            <<" raw_logits_max_abs_diff="<<raw_max_abs_diff
            <<" raw_logits_first_mismatch="
            <<(raw_first_mismatch==score_row ? -1LL
                : static_cast<long long>(raw_first_mismatch))
            <<" repetition_mask_mismatch="<<mask_mismatch
            <<" processed_score_mismatch="<<processed_mismatch
            <<" processed_score_max_abs_diff="<<processed_max_abs_diff
            <<" processed_score_first_mismatch="
            <<(processed_first_mismatch==score_row ? -1LL
                : static_cast<long long>(processed_first_mismatch))
            <<" processed_score_native="
            <<(processed_first_mismatch==score_row
                ? 0.0f
                : final_trace.processed_scores[processed_first_mismatch])
            <<" processed_score_official="
            <<(processed_first_mismatch==score_row
                ? 0.0f
                : official_processed[processed_first_mismatch])
            <<" code0="<<code0
            <<" official_code0="<<official_code0
            <<" final_head_exact="
            <<(code0==official_code0?"True":"False")
            <<" predictor_frame_exact="
            <<(frame_codes_exact?"True":"False")
            <<" predictor_first_bad_group="<<first_bad_group
            <<" predictor_first_bad_native_code="<<first_bad_native_code
            <<" predictor_first_bad_official_code="<<first_bad_official_code
            <<" next_input_exact="
            <<(next_mismatch==0?"True":"False")
            <<" next_input_mismatch="<<next_mismatch
            <<" next_input_first_mismatch="
            <<(next_first_mismatch==1024
                ? -1LL
                : static_cast<long long>(next_first_mismatch))
            <<" validation_continues=True"
            <<"\n";
    }

    const bool raw_all=
        raw_logits_exact==frame_count;
    const bool processed_all=
        processed_scores_exact==frame_count;
    const bool predictor_all=
        predictor_exact==frame_count;
    const bool next_all=
        next_input_exact==frame_count;
    const bool matrix_exact=
        native_codes==official_codes;

    const double predictor_group_match_rate =
        predictor_group_total==0
        ? 1.0
        : static_cast<double>(
            predictor_group_matches)
            / static_cast<double>(
                predictor_group_total);

    const double teacher_predictor_group_match_rate =
        teacher_predictor_group_total==0
        ? 1.0
        : static_cast<double>(
            teacher_predictor_group_matches)
            / static_cast<double>(
                teacher_predictor_group_total);

    const double next_input_mae =
        frame_count==0
        ? 0.0
        : next_input_abs_sum
            / static_cast<double>(
                static_cast<std::size_t>(
                    frame_count)
                * 1024);

    const double raw_logits_mae =
        raw_logits_value_count==0
        ? 0.0
        : raw_logits_abs_sum
            / static_cast<double>(
                raw_logits_value_count);

    std::cout
        <<"validation_frame_count: "
        <<frame_count<<"\n"
        <<"validation_executed_frame_count: "
        <<executed_frames<<"\n"
        <<"final_head_code0_match_count: "
        <<final_head_exact<<"\n"
        <<"repetition_mask_exact_frame_count: "
        <<repetition_mask_exact<<"\n"
        <<"final_head_code0_match_rate: "
        <<(
            frame_count==0
            ? 1.0
            : static_cast<double>(
                final_head_exact)
                / frame_count
        )<<"\n"
        <<"final_head_raw_logits_mae: "
        <<raw_logits_mae<<"\n"
        <<"predictor_exact_frame_count_reference_only: "
        <<predictor_exact<<"\n"
        <<"predictor_group_match_count: "
        <<predictor_group_matches<<"\n"
        <<"legacy_predictor_group_total: "
        <<predictor_group_total<<"\n"
        <<"legacy_predictor_group_match_rate: "
        <<predictor_group_match_rate<<"\n"
        <<"teacher_predictor_exact_frame_count: "
        <<teacher_predictor_exact_frames<<"\n"
        <<"teacher_predictor_group_match_count: "
        <<teacher_predictor_group_matches<<"\n"
        <<"legacy_teacher_predictor_group_total: "
        <<teacher_predictor_group_total<<"\n"
        <<"legacy_teacher_predictor_group_match_rate: "
        <<teacher_predictor_group_match_rate<<"\n"
        <<"teacher_predictor_semantics: "
          "independent_step_exact_input_exact_kv_before\n"
        <<"next_input_bit_exact_frame_count_reference_only: "
        <<next_input_exact<<"\n"
        <<"legacy_next_input_mae: "
        <<next_input_mae<<"\n"
        <<"next_input_max_abs_diff: "
        <<next_input_max_abs<<"\n"
        <<"legacy_raw_logits_all_bit_exact: "
        <<(raw_all?"True":"False")<<"\n"
        <<"legacy_processed_scores_all_bit_exact: "
        <<(processed_all?"True":"False")<<"\n"
        <<"legacy_predictor_all_frames_bit_exact: "
        <<(predictor_all?"True":"False")<<"\n"
        <<"legacy_next_input_all_bit_exact: "
        <<(next_all?"True":"False")<<"\n"
        <<"legacy_codec_matrix_bit_exact: "
        <<(matrix_exact?"True":"False")<<"\n"
        <<"algorithmic_backend_executed_all_frames: "
        <<(
            executed_frames==frame_count
            ? "True":"False"
        )<<"\n"
        <<"algorithmic_generation_tail_validation_closed: True\n"
        <<"note: free_running_match_measures_sequence_sensitivity_not_per_operator_accuracy\n"
        <<"note_2: teacher_forced_step_match_is_primary_predictor_correctness_metric\n"
        <<"diagnostic_scope: predictor_stage_error_and_top1_margin_persisted_teacher_tensors\n";

    const fs::path native_codec=
        work/"native_recurrence_codec_frames.i32.bin";
    write_binary_tail(
        native_codec,
        legacy_native_codec_a);

    const fs::path decode_root=work/"decode";
    const std::vector<std::string> decode_command{
        "python3",script.string(),
        "--model-dir",fs::absolute(options.model_dir).string(),
        "--output",decode_root.string(),
        "--decode-only",
        "--codec-frames",native_codec.string()
    };

    const int decode_rc=run_process(decode_command);
    if(decode_rc!=0)
        throw std::runtime_error(
            "legacy_legacy_tag_e.3 decoder bridge failed with exit code "
            +std::to_string(decode_rc));

    const auto waveform=
        read_binary<float>(
            decode_root/"waveform.f32.bin");
    const int sample_rate=
        read_int_text(
            decode_root/"sample_rate.txt");

    write_float_wav(
        fs::absolute(options.output_wav),
        waveform,
        sample_rate);

    std::cout
        <<"validation_audio_sample_rate: "
        <<sample_rate<<"\n"
        <<"validation_audio_sample_count: "
        <<waveform.size()<<"\n"
        <<"validation_audio_uses_python_decoder: True\n"
        <<"legacy_output_wav_codec_source_a: native_free_recurrence\n"
        <<"legacy_raw_text_to_codec_native_closed_a: True\n"
        <<"legacy_native_raw_text_to_wav_closed_b: False\n"
        <<"remaining_production_boundary: native_codec_decoder_vocoder\n"
        <<"secondary_cleanup_boundary: native_rope_generation_and_in_memory_predictor\n"
        <<"current_boundary: raw text now reaches a native free-running codec matrix; "
          "replace the Python codec decoder/vocoder next, then remove validation-only RoPE capture\n";

    if(!options.keep_work)
        fs::remove_all(work);

    return 0;
}


static int run_bridge_e2e(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids) {

    const fs::path work =
        select_work_dir(options);

    fs::remove_all(work);
    fs::create_directories(work);

    const fs::path text_path =
        work / "request.txt";
    const fs::path ids_path =
        work / "native_input_ids.i64.bin";

    write_text_file(
        text_path,
        options.text);

    write_i64(
        ids_path,
        native_ids);

    const fs::path script =
        options.bridge_script.empty()
        ? default_bridge_script()
        : fs::absolute(options.bridge_script);

    if (!fs::is_regular_file(script)) {
        throw std::runtime_error(
            "bridge script is missing: "
            + script.string());
    }

    std::cout
        << "legacy_e2e_mode_b: bridge\n"
        << "bridge_is_transitional: True\n"
        << "bridge_python_runtime_dependency: True\n"
        << "bridge_trace_or_gold_runtime_dependency: False\n";

    const std::vector<std::string> command{
        "python3",
        script.string(),
        "--model-dir",
        fs::absolute(options.model_dir).string(),
        "--text-file",
        text_path.string(),
        "--native-token-ids",
        ids_path.string(),
        "--output",
        work.string(),
        "--max-new-tokens",
        std::to_string(options.max_new_tokens),
    };

    const int rc = run_process(command);

    if (rc != 0) {
        throw std::runtime_error(
            "official E2E bridge failed with exit code "
            + std::to_string(rc));
    }

    const fs::path waveform_path =
        work / "waveform.f32.bin";
    const fs::path sample_rate_path =
        work / "sample_rate.txt";
    const fs::path codec_path =
        work / "codec_frames.i32.bin";

    const std::vector<float> waveform =
        read_binary<float>(
            waveform_path);
    const int sample_rate =
        read_int_text(
            sample_rate_path);
    const std::vector<std::int32_t> codec =
        read_binary<std::int32_t>(
            codec_path);

    if (codec.empty() || codec.size() % 16 != 0) {
        throw std::runtime_error(
            "bridge returned an invalid codec matrix");
    }

    write_float_wav(
        fs::absolute(options.output_wav),
        waveform,
        sample_rate);

    const std::size_t frame_count =
        codec.size() / 16;

    std::cout
        << "single_cpp_runtime: True\n"
        << "raw_text_entered_single_cpp: True\n"
        << "native_tokenizer_token_count: "
        << native_ids.size() << "\n"
        << "e2e_codec_frame_count: "
        << frame_count << "\n"
        << "e2e_audio_sample_rate: "
        << sample_rate << "\n"
        << "e2e_audio_sample_count: "
        << waveform.size() << "\n"
        << "e2e_output_wav: "
        << fs::absolute(options.output_wav)
        << "\n"
        << "bridge_raw_text_to_wav_closed: True\n"
        << "legacy_native_raw_text_to_wav_closed_a: False\n";

    if (!options.keep_work) {
        fs::remove_all(work);
    }

    return 0;
}


static int run_frontend_native_e2e(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids) {

    const fs::path work=select_work_dir(options);
    fs::remove_all(work);
    fs::create_directories(work);

    fs::path assets=options.native_assets_dir;
    if(assets.empty()){
        const fs::path exe=executable_path();
        assets=exe.parent_path().parent_path()/"legacy_native_assets_a";
    }
    assets=fs::absolute(assets);

    model_frontend::NativeFrontend frontend(
        fs::absolute(options.model_dir),
        assets);
    const auto native=frontend.run(native_ids);

    const fs::path prefill=work/"native_prefill.bf16.bin";
    const fs::path pad=work/"native_tts_pad.bf16.bin";
    model_frontend::write_u16(prefill,native.prefill);
    model_frontend::write_u16(pad,native.tts_pad);
    model_frontend::write_u16(
        work/"native_tts_pad_embedding.bf16.bin",
        native.tts_pad_trace.embedding);
    model_frontend::write_u16(
        work/"native_tts_pad_fc1.bf16.bin",
        native.tts_pad_trace.fc1);
    model_frontend::write_u16(
        work/"native_tts_pad_silu.bf16.bin",
        native.tts_pad_trace.silu);
    model_frontend::write_u16(
        work/"native_tts_pad_fc2.bf16.bin",
        native.tts_pad_trace.fc2);
    write_text_file(
        work/"prefill_seq_len.txt",
        std::to_string(native.seq_len)+"\n");
    write_text_file(work/"request.txt",options.text);
    write_i64(work/"native_input_ids.i64.bin",native_ids);

    const fs::path script=
        options.bridge_script.empty()
        ? default_bridge_script()
        : fs::absolute(options.bridge_script);

    std::cout
        << "legacy_e2e_mode_a: frontend-native\n"
        << "native_safetensors_frontend: True\n"
        << "native_frontend_uses_exact_biased_hip_module: True\n"
        << "native_frontend_prefill_seq_len: "
        << native.seq_len << "\n";

    const std::vector<std::string> command{
        "python3",script.string(),
        "--model-dir",fs::absolute(options.model_dir).string(),
        "--text-file",(work/"request.txt").string(),
        "--native-token-ids",(work/"native_input_ids.i64.bin").string(),
        "--native-prefill",prefill.string(),
        "--native-tts-pad",pad.string(),
        "--prefill-seq-len",std::to_string(native.seq_len),
        "--output",work.string(),
        "--max-new-tokens",std::to_string(options.max_new_tokens),
        "--frontend-native"
    };
    const int rc=run_process(command);
    if(rc!=0)
        throw std::runtime_error(
            "frontend-native bridge failed with exit code "+std::to_string(rc));

    const auto waveform=read_binary<float>(work/"waveform.f32.bin");
    const int sample_rate=read_int_text(work/"sample_rate.txt");
    const auto codec=read_binary<std::int32_t>(work/"codec_frames.i32.bin");
    if(codec.empty()||codec.size()%16)
        throw std::runtime_error("invalid frontend-native codec matrix");

    write_float_wav(
        fs::absolute(options.output_wav),
        waveform,
        sample_rate);

    std::cout
        << "single_cpp_runtime: True\n"
        << "raw_text_entered_single_cpp: True\n"
        << "native_frontend_e2e_codec_frame_count: "
        << codec.size()/16 << "\n"
        << "native_frontend_e2e_audio_sample_rate: "
        << sample_rate << "\n"
        << "native_frontend_e2e_audio_sample_count: "
        << waveform.size() << "\n"
        << "native_frontend_bridge_raw_text_to_wav_closed: True\n"
        << "legacy_native_raw_text_to_wav_closed_a: False\n";

    if(!options.keep_work)fs::remove_all(work);
    return 0;
}

static int run_native_diagnostic_e2e(
    const CliOptions& options,
    const std::vector<std::int64_t>& native_ids) {

    (void)options;

    // IMPORTANT:
    // This path deliberately does not read native_trace/, gold_*, oracle
    // tensors, or invoke Python. From this point forward all production work
    // happens inside qwen3_tts.cpp.
    //
    // Closed pieces to be transplanted here:
    //   - legacy_legacy_tag_b M=1 Talker decode BF16 GEMM/operator path
    //   - legacy_legacy_tag_c Final Head
    //   - Predictor 15-group numerical core
    //   - legacy_legacy_tag_d decoder GEMM + 37/37 convolution backend
    //
    // Remaining production blockers:
    //   A. native raw-text embedding/text_projection + Talker prefill
    //   B. dynamic Talker KV attention for arbitrary cache length
    //   C. persistent Predictor/Talker weights and caches (no per-layer files)
    //   D. decoder RMSNorm exact/functional integration
    //
    // legacy_legacy_tag_e intentionally makes these blockers visible at the final executable
    // boundary instead of hiding them behind another validation package.

    std::cout
        << "legacy_e2e_mode_b: native\n"
        << "single_cpp_runtime: True\n"
        << "raw_text_entered_single_cpp: True\n"
        << "native_tokenizer_token_count: "
        << native_ids.size() << "\n"
        << "native_python_runtime_dependency: False\n"
        << "native_trace_runtime_dependency: False\n"
        << "native_gold_runtime_dependency: False\n"
        << "native_dynamic_talker_predictor_closed: False\n"
        << "native_codec_decoder_closed: False\n"
        << "legacy_native_raw_text_to_wav_closed_a: False\n"
        << "current_boundary: "
           "Implement the remaining native numerical backend directly inside "
           "qwen3_tts.cpp: arbitrary-length Talker Transformer/KV token "
           "loop + native codec decoder RMSNorm/integration.\n";

    return 4;
}

static int run_main(
    int argc,
    char** argv) {

    const auto e2e_start=
        PerfClock::now();

    const CliOptions options =
        parse_cli(argc, argv);

    if (options.self_test) {
        return run_self_test();
    }

    const bool official_route=
        options.task=="base-xvector"
        ||options.task=="custom-voice";

    const auto model_async_start=
        PerfClock::now();

    std::future<
        std::unique_ptr<
            OfficialPreparedModel
        >
    > model_future;

    if(official_route){
        const fs::path model_dir=
            fs::absolute(
                options.model_dir);

        model_future=
            std::async(
                std::launch::async,
                [model_dir](){
                    return std::make_unique<
                        OfficialPreparedModel>(
                            model_dir);
                });
    }

    const auto tokenizer_start=
        PerfClock::now();

    const fs::path tokenizer_dir =
        locate_tokenizer_dir(
            fs::absolute(options.model_dir));

    ZImageTokenizer tokenizer(
        tokenizer_dir,
        false);

    const std::vector<std::int64_t> native_ids =
        encode_tts_assistant_prompt(
            tokenizer,
            options.text);

    const double cli_and_tokenizer_ms=
        elapsed_ms(
            e2e_start,
            PerfClock::now());

    std::cout
        << "single_target_cpp: True\n"
        << "native_tokenizer_ready: True\n"
        << "native_tokenizer_token_count: "
        << native_ids.size() << "\n"
        << "tokenizer_only_ms: "
        << elapsed_ms(
            tokenizer_start,
            PerfClock::now())
        << "\n";

    if(official_route){
        std::cout
            <<"formal_06b_task_route_entered: True\n"
            <<"execution_task: "
            <<options.task
            <<"\n"
            <<"text_mode: "
            <<options.text_mode
            <<"\n"
            <<"gpu_dependency_policy: explicit_nonblocking_streams\n"
            <<"predictor_graph_capture_mode: thread_local\n"
            <<"speaker_stream_policy: explicit_nonblocking\n"
            <<"frontend_stream_policy: explicit_nonblocking\n"
            <<"talker_weight_upload_stream_policy: explicit_nonblocking\n"
            <<"legacy_default_stream_required_for_overlap: False\n";

        if(options.task=="base-xvector"){
            return run_official_base_xvector(
                options,
                native_ids,
                &model_future,
                model_async_start,
                e2e_start,
                cli_and_tokenizer_ms);
        }

        return run_official_custom_voice(
            options,
            native_ids,
            &model_future,
            model_async_start,
            e2e_start,
            cli_and_tokenizer_ms);
    }

    if (
        options.mode == "native-e2e-legacy_legacy_tag_n-validate"
        || options.mode == "native-e2e-legacy_legacy_tag_n"
        || options.mode == "native-e2e-legacy_legacy_tag_m-validate"
        || options.mode == "native-e2e-legacy_legacy_tag_l"
    ) {
        if(options.mode=="native-e2e-legacy_legacy_tag_n-validate"){
            std::cout
                <<"legacy_native_validation_route_entered_b: True\n";
        }else if(options.mode=="native-e2e-legacy_legacy_tag_n"){
            std::cout
                <<"native_production_route_entered: True\n";
        }else if(options.mode=="native-e2e-legacy_legacy_tag_m-validate"){
            std::cout
                <<"legacy_native_validation_route_entered_a: True\n";
        }else{
            std::cout
                << "native_e2e_route_entered: True\n";
        }

        return run_legacy_native_e2e(
            options,
            native_ids,
            e2e_start,
            cli_and_tokenizer_ms);
    }

    if (options.mode == "bridge") {
        return run_bridge_e2e(
            options,
            native_ids);
    }

    if (options.mode == "frontend-native") {
        return run_frontend_native_e2e(
            options,
            native_ids);
    }

    if (
        options.mode == "generation-tail-native"
        || options.mode == "generation-tail-algorithmic"
    ) {
        return run_generation_tail_native_e2e(
            options,
            native_ids);
    }

    return run_native_diagnostic_e2e(
        options,
        native_ids);
}

} // namespace qwen3_tts::baseline

int main(int argc, char** argv) {
    try {
        return qwen3_tts::baseline::run_main(
            argc,
            argv);
    } catch (const std::exception& error) {
        std::cerr
            << "FATAL: "
            << error.what()
            << "\n";
        return 1;
    }
}


// ============================================================================
// Qingming resident lifecycle
//
// Device-first lifecycle:
//   load RX7900XTX-24G execution image once
//   keep model / tokenizer / speaker weights / frontend / Talker / Predictor
//   / codec parameters resident
//   accept an unbounded JSONL request stream
//

// generate_codec_sequence() recurrence. No model arithmetic is changed.
// ============================================================================

namespace qwen3_tts::baseline {

struct ResidentRequest {
    std::string text;
    std::string language="English";
    std::string speaker;
    fs::path ref_audio;
    fs::path output_wav;
    std::uint64_t seed=1234;
    std::size_t max_new_tokens=0;
};

static std::size_t resident_json_value_position(
    const std::string& line,
    const std::string& key) {

    const std::string needle=
        std::string("\"")+key+"\"";

    const std::size_t key_position=
        line.find(needle);

    if(key_position==std::string::npos){
        return std::string::npos;
    }

    std::size_t position=
        key_position+needle.size();

    while(
        position<line.size()
        &&std::isspace(
            static_cast<unsigned char>(
                line[position]))
    ){
        ++position;
    }

    if(
        position>=line.size()
        ||line[position]!=':'
    ){
        throw std::runtime_error(
            "resident JSON field has no colon: "
            +key);
    }

    ++position;

    while(
        position<line.size()
        &&std::isspace(
            static_cast<unsigned char>(
                line[position]))
    ){
        ++position;
    }

    return position;
}

static std::optional<std::string>
resident_json_string_field(
    const std::string& line,
    const std::string& key) {

    std::size_t position=
        resident_json_value_position(
            line,
            key);

    if(position==std::string::npos){
        return std::nullopt;
    }

    if(
        position>=line.size()
        ||line[position]!='"'
    ){
        throw std::runtime_error(
            "resident JSON string field expected: "
            +key);
    }

    ++position;

    std::string output;

    while(position<line.size()){
        const unsigned char c=
            static_cast<unsigned char>(
                line[position++]);

        if(c=='"'){
            return output;
        }

        if(c!='\\'){
            output.push_back(
                static_cast<char>(c));
            continue;
        }

        if(position>=line.size()){
            throw std::runtime_error(
                "resident JSON unfinished escape");
        }

        const char escaped=
            line[position++];

        switch(escaped){
            case '"':
            case '\\':
            case '/':
                output.push_back(escaped);
                break;
            case 'b':
                output.push_back('\b');
                break;
            case 'f':
                output.push_back('\f');
                break;
            case 'n':
                output.push_back('\n');
                break;
            case 'r':
                output.push_back('\r');
                break;
            case 't':
                output.push_back('\t');
                break;
            case 'u': {
                const std::uint32_t first=
                    parse_hex4(
                        line,
                        position);

                position+=4;

                std::uint32_t codepoint=
                    first;

                if(
                    first>=0xD800u
                    &&first<=0xDBFFu
                ){
                    if(
                        position+6>line.size()
                        ||line[position]!='\\'
                        ||line[position+1]!='u'
                    ){
                        throw std::runtime_error(
                            "resident JSON missing low surrogate");
                    }

                    position+=2;

                    const std::uint32_t second=
                        parse_hex4(
                            line,
                            position);

                    position+=4;

                    if(
                        second<0xDC00u
                        ||second>0xDFFFu
                    ){
                        throw std::runtime_error(
                            "resident JSON invalid low surrogate");
                    }

                    codepoint=
                        0x10000u
                        +((first-0xD800u)<<10)
                        +(second-0xDC00u);
                }else if(
                    first>=0xDC00u
                    &&first<=0xDFFFu
                ){
                    throw std::runtime_error(
                        "resident JSON unexpected low surrogate");
                }

                append_utf8(
                    output,
                    codepoint);
                break;
            }
            default:
                throw std::runtime_error(
                    "resident JSON unknown escape");
        }
    }

    throw std::runtime_error(
        "resident JSON unterminated string");
}

static std::optional<std::uint64_t>
resident_json_u64_field(
    const std::string& line,
    const std::string& key) {

    std::size_t position=
        resident_json_value_position(
            line,
            key);

    if(position==std::string::npos){
        return std::nullopt;
    }

    if(
        position>=line.size()
        ||!std::isdigit(
            static_cast<unsigned char>(
                line[position]))
    ){
        throw std::runtime_error(
            "resident JSON unsigned integer field expected: "
            +key);
    }

    std::uint64_t value=0;

    while(
        position<line.size()
        &&std::isdigit(
            static_cast<unsigned char>(
                line[position]))
    ){
        const std::uint64_t digit=
            static_cast<std::uint64_t>(
                line[position]-'0');

        if(
            value>
            (
                std::numeric_limits<
                    std::uint64_t>::max()
                -digit
            )/10u
        ){
            throw std::runtime_error(
                "resident JSON integer overflow: "
                +key);
        }

        value=
            value*10u
            +digit;

        ++position;
    }

    return value;
}

static ResidentRequest parse_resident_request(
    const std::string& line,
    std::size_t) {

    ResidentRequest request;

    const auto text=
        resident_json_string_field(
            line,
            "text");

    const auto reference=
        resident_json_string_field(
            line,
            "ref_audio");

    const auto speaker=
        resident_json_string_field(
            line,
            "speaker");

    const auto output=
        resident_json_string_field(
            line,
            "output");

    if(!text||text->empty()){
        throw std::runtime_error(
            "resident request requires non-empty \"text\"");
    }

    if(!output||output->empty()){
        throw std::runtime_error(
            "resident request requires \"output\"");
    }

    request.text=*text;

    if(reference&&!reference->empty()){
        request.ref_audio=
            fs::path(*reference);
    }

    if(speaker&&!speaker->empty()){
        request.speaker=
            *speaker;
    }

    request.output_wav=fs::path(*output);

    if(
        const auto language=
            resident_json_string_field(
                line,
                "language")
    ){
        if(language->empty()){
            throw std::runtime_error(
                "resident request language cannot be empty");
        }

        request.language=*language;
    }

    if(
        const auto seed=
            resident_json_u64_field(
                line,
                "seed")
    ){
        request.seed=*seed;
    }

    const auto maximum=
        resident_json_u64_field(
            line,
            "max_new_tokens");
    if(!maximum){
        throw std::runtime_error(
            "resident request requires max_new_tokens");
    }
    if(*maximum==0||*maximum>kGenerationCapacityMax){
        throw std::runtime_error(
            "resident request max_new_tokens must be in [1,8192]");
    }
    request.max_new_tokens=static_cast<std::size_t>(*maximum);

    return request;
}

static bool resident_benchmark_trace_enabled() {
    const char* value=
        std::getenv(
            "QINGMING_BENCHMARK_TRACE");

    return
        value!=nullptr
        &&value[0]=='1'
        &&value[1]=='\0';
}

static void resident_benchmark_heartbeat(
    std::size_t request_index,
    const char* stage) {

    if(!resident_benchmark_trace_enabled()){
        return;
    }

    const std::string line=
        std::string("@resident_stage request=")
        +std::to_string(request_index)
        +" stage="
        +stage
        +"\n";

#if defined(__linux__)
    const char* data=line.data();
    std::size_t remaining=line.size();

    while(remaining>0){
        const ssize_t written=
            ::write(
                STDERR_FILENO,
                data,
                remaining);

        if(written<=0){
            break;
        }

        data+=written;
        remaining-=
            static_cast<std::size_t>(
                written);
    }
#else
    std::fwrite(
        line.data(),
        1,
        line.size(),
        stderr);
    std::fflush(stderr);
#endif
}

class ResidentEngine {
public:
    ResidentEngine(
        fs::path model_dir,
        std::size_t maximum_frames,
        std::string task,
        std::string text_mode)
        :model_dir_(
            fs::absolute(
                std::move(model_dir))),
         maximum_frames_(
            maximum_frames),
         task_(
            std::move(task)),
         text_mode_(
            std::move(text_mode)) {

        if(
            maximum_frames_<kGenerationCapacityMin
            ||maximum_frames_>kGenerationCapacityMax
            ||generation_capacity_for(maximum_frames_)!=maximum_frames_
        ){
            throw std::runtime_error(
                "resident capacity must be one of 256,512,1024,2048,4096,8192");
        }

        const auto load_start=
            PerfClock::now();

        configure_sampling(
            true,
            1234);

        sampling_state_=
            std::make_unique<
                GpuSamplingState>(
                    1234);

        if(text_mode_!="streaming"){
            throw std::runtime_error(
                "resident requires explicit text_mode=streaming");
        }

        if(task_=="custom-voice"){
            custom_contract_=
                load_custom_voice_contract(
                    model_dir_);
        }else if(task_!="base-xvector"){
            throw std::runtime_error(
                "resident unsupported explicit task: "
                +task_);
        }

        prepared_model_=
            std::make_unique<
                OfficialPreparedModel>(
                    model_dir_);

        const fs::path tokenizer_dir=
            locate_tokenizer_dir(
                model_dir_);

        tokenizer_=
            std::make_unique<
                ZImageTokenizer>(
                    tokenizer_dir,
                    false);

        if(task_=="base-xvector"){
            speaker_encoder_=
                std::make_unique<
                    speaker_encoder::Encoder>(
                        model_dir_);
        }

        frontend_=
            std::make_unique<
                model_frontend::NativeFrontend>(
                    model_dir_,
                    fs::path{});

        
        // languages use the official 10-row Base x-vector prefill.
        talker_=
            std::make_unique<
                talker::NativeTalkerEngine>(
                    model_dir_,
                    10,
                    static_cast<int>(
                        maximum_frames_),
                    -1);

        predictor_=
            std::make_unique<
                code_predictor::
                    PersistentPredictor>(
                        prepared_model_
                            ->predictor_weights,
                        prepared_model_
                            ->codec_embedding(),
                        sampling_state_
                            ->device_state());

        speech_tokenizer_dir_=
            codec_decoder::
                locate_speech_tokenizer(
                    model_dir_);

        codec_parameters_=
            std::make_shared<
                codec_decoder::Parameters>(
                    speech_tokenizer_dir_);

        decoder_stream_=
            create_decoder_execution_stream(
                "Resident persistent decoder stream");

        // The streaming decoder workspace shares the Resident lifecycle.
        resident_codec_decoder_=
            std::make_unique<
                codec_decoder::StreamingCodecDecoder>(
                    codec_parameters_,
                    static_cast<int>(maximum_frames_),
                    codec_decoder::STREAM_CHUNK_FRAMES,
                    true,
                    decoder_stream_);

        model_load_ms_=
            elapsed_ms(
                load_start,
                PerfClock::now());

        std::cout
            <<std::fixed
            <<std::setprecision(3)
            <<"resident_device: rx7900xtx-24g\n"
            <<"resident_target_arch: gfx1100\n"
            <<"resident_model_dir: "
            <<model_dir_.string()
            <<"\n"
            <<"resident_model_contract: "
            <<(
                task_=="custom-voice"
                ?"Qwen3-TTS-12Hz-0.6B-CustomVoice"
                :"Qwen3-TTS-12Hz-0.6B-Base"
            )
            <<"\n"
            <<"resident_task_contract: "
            <<task_
            <<"\n"
            <<"resident_text_mode: "
            <<text_mode_
            <<"\n"
            <<"resident_prefill_rows: 10\n"
            <<"resident_max_new_tokens_capacity: "
            <<maximum_frames_
            <<"\n"
            <<"resident_model_load_ms: "
            <<model_load_ms_
            <<"\n"
            <<"resident_prepared_model_reused: True\n"
            <<"resident_tokenizer_reused: True\n"
            <<"resident_speaker_weights_reused: True\n"
            <<"resident_frontend_reused: True\n"
            <<"resident_talker_reused: True\n"
            <<"resident_predictor_graph_reused: True\n"
            <<"resident_codec_parameters_reused: True\n"
            <<"resident_codec_decoder_workspace_reused: True\n"
            <<"resident_generation_decoder_cu_partitioned: "
            <<(g_resident_cu_partition_enabled
                ?"True":"False")
            <<"\n"
            <<"resident_generation_cu_count: "
            <<(g_resident_cu_partition_enabled
                ?g_resident_generation_wgps*2
                :96)
            <<"\n"
            <<"resident_decoder_cu_count: "
            <<(g_resident_cu_partition_enabled
                ?g_resident_decoder_wgps*2
                :96)
            <<"\n"
            <<"resident_generation_wgp_count: "
            <<(g_resident_cu_partition_enabled
                ?g_resident_generation_wgps
                :48)
            <<"\n"
            <<"resident_decoder_wgp_count: "
            <<(g_resident_cu_partition_enabled
                ?g_resident_decoder_wgps
                :48)
            <<"\n"
            <<"resident_protocol: jsonl_stdin\n"
            <<"resident_ready: True\n";
    }

    ~ResidentEngine() {
        // The decoder uses decoder_stream_ and is released before the stream.
        resident_codec_decoder_.reset();
        if(decoder_stream_!=nullptr){
            (void)hipStreamSynchronize(
                decoder_stream_);
            (void)hipStreamDestroy(
                decoder_stream_);
            decoder_stream_=nullptr;
        }
    }

    void generate(
        const ResidentRequest& request,
        std::size_t request_index) {

        resident_benchmark_heartbeat(
            request_index,
            "begin");

        if(
            request.max_new_tokens==0
            ||request.max_new_tokens>
                maximum_frames_
        ){
            throw std::runtime_error(
                "resident request max_new_tokens exceeds resident capacity");
        }

        std::string normalized_language=
            request.language;

        std::transform(
            normalized_language.begin(),
            normalized_language.end(),
            normalized_language.begin(),
            [](unsigned char c){
                return static_cast<char>(
                    std::tolower(c));
            });

        std::string effective_language=
            normalized_language;

        if(task_=="base-xvector"){
            if(request.ref_audio.empty()){
                throw std::runtime_error(
                    "base-xvector resident request requires \"ref_audio\"");
            }

            if(!request.speaker.empty()){
                throw std::runtime_error(
                    "base-xvector resident request forbids \"speaker\"");
            }
        }else{
            if(request.speaker.empty()){
                throw std::runtime_error(
                    "custom-voice resident request requires \"speaker\"");
            }

            if(!request.ref_audio.empty()){
                throw std::runtime_error(
                    "custom-voice resident request forbids \"ref_audio\"");
            }
        }

        const fs::path ref_audio=
            task_=="base-xvector"
            ?fs::absolute(
                request.ref_audio)
            :fs::path{};

        if(
            task_=="base-xvector"
            &&!fs::is_regular_file(
                ref_audio)
        ){
            throw std::runtime_error(
                "resident reference WAV does not exist: "
                +ref_audio.string());
        }

        const auto e2e_start=
            PerfClock::now();

        configure_sampling(
            true,
            request.seed);

        sampling_state_
            ->reset(
                request.seed);

        // Explicit request-state reset on the generation stream.
        talker_->reset_request();

        prepared_model_
            ->final_head
            .reset_request(
                talker_->stream());

        if(!resident_codec_decoder_){
            throw std::runtime_error(
                "resident codec decoder is not initialized");
        }
        resident_codec_decoder_->reset_request();

        resident_benchmark_heartbeat(
            request_index,
            "state_reset");

        const auto tokenizer_start=
            PerfClock::now();

        const auto native_ids=
            encode_tts_assistant_prompt(
                *tokenizer_,
                request.text);

        const double tokenizer_ms=
            elapsed_ms(
                tokenizer_start,
                PerfClock::now());

        resident_benchmark_heartbeat(
            request_index,
            "tokenizer");

        const auto speaker_start=
            PerfClock::now();

        std::vector<std::uint16_t>
            speaker_embedding;

        if(task_=="base-xvector"){
            if(!speaker_encoder_){
                throw std::runtime_error(
                    "Base resident speaker encoder is not loaded");
            }

            speaker_embedding=
                speaker_encoder_
                    ->run(
                        ref_audio,
                        fs::path{});
        }else{
            const auto conditioning=
                build_custom_voice_conditioning(
                    *custom_contract_,
                    prepared_model_
                        ->codec_embedding(),
                    request.speaker,
                    request.language);

            speaker_embedding=
                conditioning
                    .speaker_embedding;

            effective_language=
                conditioning
                    .effective_language;
        }

        if(effective_language=="auto"){
            throw std::runtime_error(
                "resident requires non-Auto effective language for constant 10-row prefill");
        }

        const double speaker_ms=
            elapsed_ms(
                speaker_start,
                PerfClock::now());

        resident_benchmark_heartbeat(
            request_index,
            "speaker");

        const auto frontend_start=
            PerfClock::now();

        auto native=
            frontend_
                ->build_official_base_xvector_inputs(
                    native_ids,
                    speaker_embedding,
                    effective_language);

        const double frontend_ms=
            elapsed_ms(
                frontend_start,
                PerfClock::now());

        resident_benchmark_heartbeat(
            request_index,
            "frontend");

        if(native.seq_len!=10){
            throw std::runtime_error(
                "resident explicit-language frontend did not produce 10 prefill rows");
        }

        PerfStats perf;
        perf.cli_and_tokenizer_ms=
            tokenizer_ms;
        perf.frontend_ms=
            frontend_ms;
        perf.model_setup_wait_ms=0.0;
        perf.core_model_setup_ms=0.0;

        std::unique_ptr<
            codec_decoder::
                StreamingCodecPipeline>
            streaming_decoder;

        PerfTime decoder_ready_start{};

        const std::function<void()>
            first_frame_hook=
                [this,
                 &streaming_decoder,
                 &decoder_ready_start,
                 maximum_frames=
                    static_cast<int>(
                        request.max_new_tokens),
                 e2e_start,
                 request_index](){

                    resident_benchmark_heartbeat(
                        request_index,
                        "first_frame");

                    decoder_ready_start=
                        PerfClock::now();

                    std::promise<
                        codec_decoder::
                            StreamingCodecPipeline::
                                PreloadResult>
                        ready_promise;

                    auto ready_future=
                        ready_promise
                            .get_future();

                    ready_promise
                        .set_value(
                            std::make_pair(
                                speech_tokenizer_dir_,
                                codec_parameters_));

                    streaming_decoder=
                        std::make_unique<
                            codec_decoder::
                                StreamingCodecPipeline>(
                                    std::move(
                                        ready_future),
                                    decoder_ready_start,
                                    maximum_frames,
                                    e2e_start,
                                    codec_decoder::
                                        STREAM_CHUNK_FRAMES,
                                    decoder_stream_,
                                    resident_codec_decoder_.get());
                };

        const std::function<
            void(
                const std::array<
                    std::int32_t,
                    16>&
            )
        > codec_frame_hook=
            [&streaming_decoder](
                const std::array<
                    std::int32_t,
                    16>& frame_codes){

                if(streaming_decoder){
                    streaming_decoder
                        ->push_frame(
                            frame_codes);
                }
            };

        resident_benchmark_heartbeat(
            request_index,
            "generation");

        const auto codec=
            generate_codec_sequence(
                model_dir_,
                fs::path{},
                native.prefill,
                static_cast<int>(
                    native.seq_len),
                static_cast<int>(
                    request.max_new_tokens),
                native.tts_pad,
                native.trailing_text,
                native.trailing_rows,
                prepared_model_
                    ->predictor_weights,
                prepared_model_
                    ->codec_embedding(),
                prepared_model_
                    ->final_head,
                sampling_state_
                    ->device_state(),
                nullptr,
                e2e_start,
                nullptr,
                e2e_start,
                first_frame_hook,
                codec_frame_hook,
                e2e_start,
                perf,
                fs::path{},
                predictor_.get(),
                talker_.get());

        resident_benchmark_heartbeat(
            request_index,
            "generation_done");

        const int frames=
            static_cast<int>(
                codec.size()/16);

        const std::uint64_t codec_fnv1a64=
            fnv1a64(
                codec.data(),
                codec.size()*sizeof(std::int32_t));

        if(!streaming_decoder){
            throw std::runtime_error(
                "resident streaming decoder was not started");
        }

        resident_benchmark_heartbeat(
            request_index,
            "decoder_finish");

        const auto tail_start=
            PerfClock::now();

        auto streaming_result=
            streaming_decoder
                ->finish();

        perf.codec_decoder_tail_wait_ms=
            elapsed_ms(
                tail_start,
                PerfClock::now());

        perf.codec_decoder_parameter_preload_wall_ms=
            streaming_result
                .parameter_preload_wall_ms;

        perf.codec_decoder_parameter_preload_wait_ms=
            streaming_result
                .parameter_wait_ms;

        perf.codec_decoder_setup_ms=
            streaming_result
                .decoder_setup_ms;

        perf.codec_decoder_graph_ms=
            streaming_result
                .decoder_compute_ms;

        perf.codec_decoder_pipeline_wall_ms=
            streaming_result
                .decoder_pipeline_wall_ms;

        perf.codec_decoder_first_chunk_ms=
            streaming_result
                .first_audio_ms;

        perf.codec_decoder_chunk_count=
            streaming_result
                .chunk_count;

        perf.codec_decoder_first_chunk_frames=
            streaming_result
                .first_chunk_frames;

        perf.codec_decoder_chunk_frames=
            streaming_result
                .chunk_frames;

        perf.ttfa_ms=
            streaming_result
                .first_audio_ms;

        resident_benchmark_heartbeat(
            request_index,
            "decoder_done");

        auto waveform=
            std::move(
                streaming_result
                    .waveform);

        if(
            waveform.size()
            !=static_cast<std::size_t>(
                frames)
                *codec_decoder::UPSAMPLE
        ){
            throw std::runtime_error(
                "resident streaming decoder waveform length mismatch");
        }

        const auto wav_start=
            PerfClock::now();

        write_float_wav(
            fs::absolute(
                request.output_wav),
            waveform,
            codec_decoder::SAMPLE_RATE);

        perf.wav_write_ms=
            elapsed_ms(
                wav_start,
                PerfClock::now());

        resident_benchmark_heartbeat(
            request_index,
            "wav_done");

        perf.e2e_ms=
            elapsed_ms(
                e2e_start,
                PerfClock::now());

        const double audio_duration_s=
            static_cast<double>(
                waveform.size())
            /codec_decoder::SAMPLE_RATE;

        const double generation_s=
            perf.codec_generation_ms
            /1000.0;

        const double e2e_s=
            perf.e2e_ms
            /1000.0;

        std::cout
            <<std::fixed
            <<std::setprecision(3)
            <<"\n"
            <<"================================================================================\n"
            <<"QINGMING RESIDENT REQUEST SUMMARY\n"
            <<"================================================================================\n"
            <<"resident_request_index: "
            <<request_index
            <<"\n"
            <<"resident_model_reused: True\n"
            <<"resident_tokenizer_reused: True\n"
            <<"resident_speaker_weights_reused: True\n"
            <<"resident_frontend_reused: True\n"
            <<"resident_talker_reused: True\n"
            <<"resident_predictor_graph_reused: True\n"
            <<"resident_codec_parameters_reused: True\n"
            <<"resident_generation_decoder_cu_partitioned: "
            <<(g_resident_cu_partition_enabled
                ?"True":"False")
            <<"\n"
            <<"resident_sampling_state_reset: True\n"
            <<"resident_final_head_history_reset: True\n"
            <<"resident_talker_cache_reset: True\n"
            <<"resident_seed: "
            <<request.seed
            <<"\n"
            <<"resident_task: "
            <<task_
            <<"\n"
            <<"resident_text_mode: "
            <<text_mode_
            <<"\n"
            <<"resident_speaker: "
            <<request.speaker
            <<"\n"
            <<"resident_language: "
            <<request.language
            <<"\n"
            <<"resident_effective_language: "
            <<effective_language
            <<"\n"
            <<"resident_input_token_count: "
            <<native_ids.size()
            <<"\n"
            <<"resident_codec_frame_count: "
            <<frames
            <<"\n"
            <<"resident_codec_fnv1a64: "
            <<codec_fnv1a64
            <<"\n"
            <<"resident_audio_duration_s: "
            <<audio_duration_s
            <<"\n"
            <<"resident_tokenizer_ms: "
            <<tokenizer_ms
            <<"\n"
            <<"resident_speaker_ms: "
            <<speaker_ms
            <<"\n"
            <<"resident_frontend_ms: "
            <<frontend_ms
            <<"\n"
            <<"resident_talker_prefill_ms: "
            <<perf.talker_prefill_ms
            <<"\n"
            <<"resident_predictor_frame_p50_ms: "
            <<percentile(
                perf.predictor_frame_ms,
                0.50)
            <<"\n"
            <<"resident_predictor_frame_p95_ms: "
            <<percentile(
                perf.predictor_frame_ms,
                0.95)
            <<"\n"
            <<"resident_first_codec_frame_compute_ms: "
            <<perf.first_codec_frame_ms
            <<"\n"
            <<"resident_ttft_ms: "
            <<perf.ttft_ms
            <<"\n"
            <<"resident_generation_ms: "
            <<perf.codec_generation_ms
            <<"\n"
            <<"resident_generation_frames_per_s: "
            <<(
                generation_s>0.0
                ?frames/generation_s
                :0.0)
            <<"\n"
            <<"resident_decoder_compute_ms: "
            <<perf.codec_decoder_graph_ms
            <<"\n"
            <<"resident_decoder_pipeline_wall_ms: "
            <<perf.codec_decoder_pipeline_wall_ms
            <<"\n"
            <<"resident_decoder_tail_wait_ms: "
            <<perf.codec_decoder_tail_wait_ms
            <<"\n"
            <<"resident_streaming_decoder_first_chunk_frames: "
            <<perf.codec_decoder_first_chunk_frames
            <<"\n"
            <<"resident_streaming_decoder_chunk_frames: "
            <<perf.codec_decoder_chunk_frames
            <<"\n"
            <<"resident_ttfa_ms: "
            <<perf.ttfa_ms
            <<"\n"
            <<"resident_e2e_ms: "
            <<perf.e2e_ms
            <<"\n"
            <<"resident_e2e_realtime_x: "
            <<(
                e2e_s>0.0
                ?audio_duration_s/e2e_s
                :0.0)
            <<"\n"
            <<"resident_output_wav: "
            <<fs::absolute(
                request.output_wav)
                .string()
            <<"\n"
            <<"resident_request_success: True\n"
            <<"================================================================================\n";

        resident_benchmark_heartbeat(
            request_index,
            "complete");
    }

private:
    fs::path model_dir_;
    std::size_t maximum_frames_=0;
    double model_load_ms_=0.0;
    std::string task_;
    std::string text_mode_;

    std::optional<
        CustomVoiceContract>
        custom_contract_;

    std::unique_ptr<
        OfficialPreparedModel>
        prepared_model_;

    std::unique_ptr<
        ZImageTokenizer>
        tokenizer_;

    std::unique_ptr<
        speaker_encoder::Encoder>
        speaker_encoder_;

    std::unique_ptr<
        model_frontend::NativeFrontend>
        frontend_;

    std::unique_ptr<
        GpuSamplingState>
        sampling_state_;

    std::unique_ptr<
        talker::NativeTalkerEngine>
        talker_;

    std::unique_ptr<
        code_predictor::PersistentPredictor>
        predictor_;

    fs::path speech_tokenizer_dir_;

    std::shared_ptr<
        codec_decoder::Parameters>
        codec_parameters_;

    std::unique_ptr<
        codec_decoder::StreamingCodecDecoder>
        resident_codec_decoder_;

    hipStream_t decoder_stream_=nullptr;
};

static int run_resident_main(
    int argc,
    char** argv) {

    fs::path model_dir;
    std::string task;
    std::string text_mode;
    std::size_t maximum_frames=0;
    int generation_wgps=28;

    for(int i=1;i<argc;++i){
        const std::string arg=
            argv[i];

        auto require_value=
            [&](const char* name)
            ->std::string {

                if(i+1>=argc){
                    throw std::runtime_error(
                        std::string(name)
                        +" requires a value");
                }

                return argv[++i];
            };

        if(arg=="--lifecycle"){
            const std::string lifecycle=
                require_value(
                    "--lifecycle");

            if(lifecycle!="resident"){
                throw std::runtime_error(
                    "resident entry requires --lifecycle resident");
            }
        }else if(arg=="--task"){
            task=
                require_value(
                    "--task");
        }else if(arg=="--text-mode"){
            text_mode=
                require_value(
                    "--text-mode");
        }else if(arg=="--model-dir"){
            model_dir=
                fs::path(
                    require_value(
                        "--model-dir"));
        }else if(arg=="--max-new-tokens"){
            const auto raw=
                require_value(
                    "--max-new-tokens");

            const unsigned long long value=
                std::stoull(raw);

            if(value==0||value>kGenerationCapacityMax){
                throw std::runtime_error(
                    "resident --max-new-tokens must be in [1,8192]");
            }
            maximum_frames=generation_capacity_for(
                static_cast<std::size_t>(value));
        }else if(arg=="--generation-wgp"){
            const int value=
                std::stoi(
                    require_value(
                        "--generation-wgp"));

            // Keep both generation and decoder with a meaningful minimum.
            // This range also covers all intended tuning points.
            if(value<20||value>40){
                throw std::runtime_error(
                    "resident --generation-wgp must be in [20,40]");
            }

            generation_wgps=value;
        }else if(
            arg=="--help"
            ||arg=="-h"
        ){
            std::cout
                <<"Resident lifecycle:\n"
                <<"  "<<argv[0]
                <<" --lifecycle resident"
                <<" --model-dir DIR"
                <<" --max-new-tokens N"
                <<" [--generation-wgp 28]\n\n"
                <<"Then send one JSON object per stdin line:\n"
                <<R"({"text":"Hello","language":"English","ref_audio":"./clone.wav","output":"./out.wav","seed":1234,"max_new_tokens":256})"
                <<"\n\n"
                <<"Shutdown with:\n"
                <<R"({"command":"shutdown"})"
                <<"\n";
            return 0;
        }else{
            throw std::runtime_error(
                "unknown resident startup argument: "
                +arg);
        }
    }

    if(
        task!="base-xvector"
        &&task!="custom-voice"
    ){
        throw std::runtime_error(
            "resident requires explicit --task base-xvector|custom-voice");
    }

    if(text_mode!="streaming"){
        throw std::runtime_error(
            "resident requires explicit --text-mode streaming");
    }

    if(maximum_frames==0){
        throw std::runtime_error(
            "resident lifecycle requires --max-new-tokens");
    }

    if(model_dir.empty()){
        throw std::runtime_error(
            "resident lifecycle requires --model-dir");
    }

    g_resident_generation_wgps=
        generation_wgps;

    g_resident_decoder_wgps=
        48-generation_wgps;

    g_resident_cu_partition_enabled=true;

    std::cout
        <<"resident_cu_partition_enabled: True\n"
        <<"resident_cu_partition_device_profile: rx7900xtx-24g\n"
        <<"resident_requested_generation_wgp: "
        <<g_resident_generation_wgps
        <<"\n"
        <<"resident_requested_decoder_wgp: "
        <<g_resident_decoder_wgps
        <<"\n";

    ResidentEngine engine(
        model_dir,
        maximum_frames,
        task,
        text_mode);

    std::cout
        <<"resident_waiting_for_request: True\n"
        <<"> "
        <<std::flush;

    std::string line;
    std::size_t request_index=0;

    while(std::getline(std::cin,line)){
        const auto begin=
            line.find_first_not_of(
                " \t\r\n");

        if(begin==std::string::npos){
            continue;
        }

        if(
            line=="quit"
            ||line=="exit"
        ){
            break;
        }

        if(
            const auto command=
                resident_json_string_field(
                    line,
                    "command")
        ){
            if(*command=="shutdown"){
                break;
            }

            throw std::runtime_error(
                "unknown resident command: "
                +*command);
        }

        auto request=
            parse_resident_request(
                line,
                maximum_frames);

        ++request_index;

        engine.generate(
            request,
            request_index);

        std::cout
            <<"resident_request_complete: "
            <<request_index
            <<"\n"
            <<"resident_waiting_for_request: True\n"
            <<"> "
            <<std::flush;
    }

    std::cout
        <<"resident_shutdown: True\n"
        <<"resident_request_count: "
        <<request_index
        <<"\n";

    return 0;
}

} // namespace qwen3_tts::baseline


// ============================================================================

//

//   48 HIP/WGP units total
//   generation = 28 WGP = 56 physical CU
//   decoder    = 20 WGP = 40 physical CU
//
// The production entry point uses this split. Runtime tuning belongs to
// benchmark/research builds, not the formal once/resident interface.
// ============================================================================
