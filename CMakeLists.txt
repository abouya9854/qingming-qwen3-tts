cmake_minimum_required(VERSION 3.20)

project(qingming_qwen3_tts NONE)

set(QINGMING_DEVICE "rx7900xtx-24g" CACHE STRING "Device backend")
set(QINGMING_MODEL_FAMILY "" CACHE STRING "Execution family")

if(QINGMING_MODEL_FAMILY STREQUAL "")
    message(FATAL_ERROR "QINGMING_MODEL_FAMILY is required")
endif()

if(QINGMING_DEVICE STREQUAL "rx7900xtx-24g")
    set(ROCM_PATH "/opt/rocm-7.2.4" CACHE PATH "ROCm installation")
    set(HIP_CLANG "${ROCM_PATH}/lib/llvm/bin/clang++" CACHE FILEPATH "HIP clang++")
    set(HOST_CXX "${ROCM_PATH}/lib/llvm/bin/clang++" CACHE FILEPATH "Host C++ compiler")

    if(QINGMING_MODEL_FAMILY STREQUAL "1.7b")
        set(FAMILY_DEFINE "QINGMING_MODEL_FAMILY_1_7B=1")
        set(FAMILY_SOURCE "${CMAKE_SOURCE_DIR}/devices/rx7900xtx-24g/qwen3_tts_1_7b.cpp")
    elseif(QINGMING_MODEL_FAMILY STREQUAL "0.6b")
        set(FAMILY_DEFINE "QINGMING_MODEL_FAMILY_0_6B=1")
        set(FAMILY_SOURCE "${CMAKE_SOURCE_DIR}/devices/rx7900xtx-24g/qwen3_tts_0_6b.cpp")
    else()
        message(FATAL_ERROR "RX7900XTX supports QINGMING_MODEL_FAMILY=0.6b or 1.7b")
    endif()

    set(PROD_NAME "qingming-qwen3-tts_rx7900xtx-24g_${QINGMING_MODEL_FAMILY}")
    set(BENCH_NAME "qingming-qwen3-tts_rx7900xtx-24g_benchmark")
    set(PROD_BINARY "${CMAKE_BINARY_DIR}/${PROD_NAME}")
    set(BENCH_BINARY "${CMAKE_BINARY_DIR}/${BENCH_NAME}")

    add_custom_command(OUTPUT "${PROD_BINARY}"
        COMMAND "${HIP_CLANG}" -x hip --offload-arch=gfx1100 "--rocm-path=${ROCM_PATH}" "-D${FAMILY_DEFINE}"
                -std=c++20 -O3 -DNDEBUG -pthread -Wall -Wextra -Wpedantic -Wno-unused-function -Wno-unused-result
                "${CMAKE_SOURCE_DIR}/main.cpp" -o "${PROD_BINARY}"
        DEPENDS "${CMAKE_SOURCE_DIR}/main.cpp" "${CMAKE_SOURCE_DIR}/devices/rx7900xtx-24g/device.h" "${FAMILY_SOURCE}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" VERBATIM)

    add_custom_command(OUTPUT "${BENCH_BINARY}"
        COMMAND "${HOST_CXX}" -std=c++20 -O3 -DNDEBUG -pthread -Wall -Wextra -Wpedantic
                "${CMAKE_SOURCE_DIR}/devices/rx7900xtx-24g/benchmark.cpp" -o "${BENCH_BINARY}"
        DEPENDS "${CMAKE_SOURCE_DIR}/devices/rx7900xtx-24g/benchmark.cpp"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" VERBATIM)

elseif(QINGMING_DEVICE STREQUAL "rtx4090-24g")
    set(CUDA_PATH "/usr/local/cuda" CACHE PATH "CUDA Toolkit installation")
    set(NVCC "${CUDA_PATH}/bin/nvcc" CACHE FILEPATH "NVIDIA CUDA compiler")
    set(HOST_CXX "/usr/bin/g++" CACHE FILEPATH "Host C++ compiler")

    if(NOT EXISTS "${NVCC}")
        message(FATAL_ERROR "nvcc not found at ${NVCC}; set CUDA_PATH or NVCC explicitly")
    endif()

    if(NOT EXISTS "${HOST_CXX}")
        message(FATAL_ERROR "host C++ compiler not found at ${HOST_CXX}")
    endif()

    if(QINGMING_MODEL_FAMILY STREQUAL "1.7b")
        set(FAMILY_DEFINE "QINGMING_MODEL_FAMILY_1_7B=1")
        set(FAMILY_SOURCE "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/qwen3_tts_1_7b.cu")
    elseif(QINGMING_MODEL_FAMILY STREQUAL "0.6b")
        set(FAMILY_DEFINE "QINGMING_MODEL_FAMILY_0_6B=1")
        set(FAMILY_SOURCE "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/qwen3_tts_0_6b.cu")
    else()
        message(FATAL_ERROR "RTX4090 supports QINGMING_MODEL_FAMILY=0.6b or 1.7b")
    endif()

    set(PROD_NAME "qingming-qwen3-tts_rtx4090-24g_${QINGMING_MODEL_FAMILY}")
    set(BENCH_NAME "qingming-qwen3-tts_rtx4090-24g_benchmark")
    set(PROD_BINARY "${CMAKE_BINARY_DIR}/${PROD_NAME}")
    set(BENCH_BINARY "${CMAKE_BINARY_DIR}/${BENCH_NAME}")

    add_custom_command(OUTPUT "${PROD_BINARY}"
        COMMAND "${NVCC}" -arch=sm_89 "-D${FAMILY_DEFINE}" -std=c++20 -O3 -DNDEBUG
                -Xcompiler=-pthread,-Wall,-Wextra
                "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/main.cu" -o "${PROD_BINARY}"
        DEPENDS "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/main.cu" "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/device.h" "${FAMILY_SOURCE}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" VERBATIM)

    add_custom_command(OUTPUT "${BENCH_BINARY}"
        COMMAND "${HOST_CXX}" -std=c++20 -O3 -DNDEBUG -pthread -Wall -Wextra -Wpedantic
                "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/benchmark.cpp" -o "${BENCH_BINARY}"
        DEPENDS "${CMAKE_SOURCE_DIR}/devices/rtx4090-24g/benchmark.cpp"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" VERBATIM)
else()
    message(FATAL_ERROR "Unsupported QINGMING_DEVICE=${QINGMING_DEVICE}")
endif()

set(PROD_TARGET "qingming_${QINGMING_DEVICE}_${QINGMING_MODEL_FAMILY}_image")
set(BENCH_TARGET "qingming_${QINGMING_DEVICE}_benchmark_host")
add_custom_target("${PROD_TARGET}" ALL DEPENDS "${PROD_BINARY}")
add_custom_target("${BENCH_TARGET}" ALL DEPENDS "${BENCH_BINARY}")
add_dependencies("${BENCH_TARGET}" "${PROD_TARGET}")
