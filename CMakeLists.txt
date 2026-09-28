cmake_minimum_required(VERSION 3.16)
project(memfabric_matmul_allreduce LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()

if(NOT DEFINED MEMFABRIC_HYBRID_HOME_PATH)
  set(MEMFABRIC_HYBRID_HOME_PATH "$ENV{MEMFABRIC_HYBRID_HOME_PATH}")
endif()
if(MEMFABRIC_HYBRID_HOME_PATH STREQUAL "")
  message(FATAL_ERROR "MEMFABRIC_HYBRID_HOME_PATH is required to build the operator")
endif()

# V1 is deliberately pinned to Ascend 310P3. The registry-invoke packaging
# integration is kept in one place so CANN 9.1 generated/official build glue
# can replace this bootstrap without touching operator/runtime sources.
set(MFMC2_SOC "ascend310p3" CACHE STRING "Target SoC")
set(MFMC2_VENDOR "memfabric" CACHE STRING "Custom OPP vendor name")

add_library(memfabric_mc2_runtime STATIC
  runtime/memfabric_runtime.cpp
)
target_include_directories(memfabric_mc2_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
)

# Host registration and kernel compilation are wired by the CANN 9.1
# registry-invoke toolchain on the target build machine. Keep source lists
# explicit for deterministic packaging and static review.
set(MFMC2_OP_HOST_SOURCES
  op_host/memfabric_matmul_allreduce_def.cpp
  op_host/memfabric_matmul_allreduce_tiling.cpp
)
set(MFMC2_OP_KERNEL_SOURCES
  op_kernel/memfabric_matmul_allreduce.cpp
)

add_custom_target(memfabric_mc2_sources ALL
  SOURCES ${MFMC2_OP_HOST_SOURCES} ${MFMC2_OP_KERNEL_SOURCES})

message(STATUS "MemFabric MC2 bootstrap configured for ${MFMC2_SOC}")
message(STATUS "CANN: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "MemFabric SDK: ${MEMFABRIC_HYBRID_HOME_PATH}")
