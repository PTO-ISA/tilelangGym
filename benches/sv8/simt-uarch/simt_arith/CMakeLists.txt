# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
#
# Standalone ST:
#   # A5 (default)
#   cmake -DPTO_ARCH=a5 -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 -DPTO_ISA_ROOT=/path/to/pto-isa ..
#   # A6
#   cmake -DPTO_ARCH=a6 -DRUN_MODE=sim -DSOC_VERSION=dav_9201 -DPTO_ISA_ROOT=/path/to/pto-isa ..
#   make -j && python3 ../gen_data.py && ./bin/simt_add

cmake_minimum_required(VERSION 3.16)
project(simt_arith_test)

# a5: dav-c310-vec + Ascend950PR_9599
# a6: dav-920r1-vec + dav_9201
set(PTO_ARCH "a5" CACHE STRING "Target NPU generation: a5 or a6")
set_property(CACHE PTO_ARCH PROPERTY STRINGS a5 a6)

if(PTO_ARCH STREQUAL "a5")
    set(CCE_AICORE_ARCH "dav-c310-vec")
    set(PTO_ARCH_DEFAULT_SOC "Ascend950PR_9599")
elseif(PTO_ARCH STREQUAL "a6")
    set(CCE_AICORE_ARCH "dav-920r1-vec")
    set(PTO_ARCH_DEFAULT_SOC "dav_9201")
else()
    message(FATAL_ERROR "Unsupported PTO_ARCH=${PTO_ARCH}; use a5 or a6")
endif()

if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
    set(SOC_VERSION "${PTO_ARCH_DEFAULT_SOC}" CACHE STRING "Simulator / SoC version" FORCE)
endif()
if(NOT DEFINED RUN_MODE OR RUN_MODE STREQUAL "")
    set(RUN_MODE "sim" CACHE STRING "Runtime: sim or npu")
endif()

message(STATUS "simt_arith: PTO_ARCH=${PTO_ARCH} CCE_AICORE_ARCH=${CCE_AICORE_ARCH} SOC_VERSION=${SOC_VERSION} RUN_MODE=${RUN_MODE}")

set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED PTO_ISA_ROOT)
    set(PTO_ISA_ROOT "/mnt/fluxdata/onasismelchiorjeyakumar/workspace/pto-isa")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

find_package(GTest REQUIRED)

set(PTO_GLIBCXX_USE_CXX11_ABI "auto" CACHE STRING "Force _GLIBCXX_USE_CXX11_ABI (auto/0/1) for linking with system libs such as GTest.")
set_property(CACHE PTO_GLIBCXX_USE_CXX11_ABI PROPERTY STRINGS auto 0 1)
if(PTO_GLIBCXX_USE_CXX11_ABI STREQUAL "auto")
    if(UNIX AND NOT APPLE AND TARGET GTest::gtest AND (CMAKE_NM OR EXISTS "/usr/bin/nm"))
        get_target_property(_pto_gtest_loc GTest::gtest IMPORTED_LOCATION)
        if(NOT _pto_gtest_loc)
            get_target_property(_pto_gtest_loc GTest::gtest LOCATION)
        endif()
        if(_pto_gtest_loc AND EXISTS "${_pto_gtest_loc}")
            execute_process(
                COMMAND bash -lc "nm -C \"${_pto_gtest_loc}\" 2>/dev/null | grep -q 'GetBoolAssertionFailureMessage\\[abi:cxx11\\]'"
                RESULT_VARIABLE _pto_gtest_is_abi1
            )
            if(NOT _pto_gtest_is_abi1 EQUAL 0)
                set(PTO_GLIBCXX_USE_CXX11_ABI "0" CACHE STRING "" FORCE)
            endif()
        endif()
        unset(_pto_gtest_loc)
        unset(_pto_gtest_is_abi1)
    endif()
endif()
if(PTO_GLIBCXX_USE_CXX11_ABI STREQUAL "0")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes
    -fstack-protector-strong
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PTO_ISA_ROOT}/include
    ${PTO_ISA_ROOT}/tests/common
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

function(pto_vec_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    # A5: --cce-aicore-arch=dav-c310-vec
    # A6: --cce-aicore-arch=dav-920r1-vec
    target_compile_options(${NAME}_kernel PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=${CCE_AICORE_ARCH}
        -DREGISTER_BASE
        -std=c++17
    )
    if(NOT DEFINED SIMT_THREAD_X)
        set(SIMT_THREAD_X 1024)
    endif()
    if(NOT DEFINED SIMT_THREAD_Y)
        set(SIMT_THREAD_Y 1)
    endif()
    if(PTO_ARCH STREQUAL "a6")
        target_compile_definitions(${NAME}_kernel PRIVATE PTO_NPU_A6=1 SIMT_THREAD_X=${SIMT_THREAD_X} SIMT_THREAD_Y=${SIMT_THREAD_Y})
    else()
        target_compile_definitions(${NAME}_kernel PRIVATE PTO_NPU_A5=1 SIMT_THREAD_X=${SIMT_THREAD_X} SIMT_THREAD_Y=${SIMT_THREAD_Y})
    endif()

    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PTO_ISA_ROOT}/tests/common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()

pto_vec_st(simt_arith)
