cmake_minimum_required(VERSION 3.19.0)
project(memfabric_matmul_allreduce LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR ASCEND_CANN_PACKAGE_PATH STREQUAL "")
  if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()

if(NOT DEFINED MEMFABRIC_HYBRID_HOME_PATH OR MEMFABRIC_HYBRID_HOME_PATH STREQUAL "")
  set(MEMFABRIC_HYBRID_HOME_PATH "$ENV{MEMFABRIC_HYBRID_HOME_PATH}")
endif()
if(MEMFABRIC_HYBRID_HOME_PATH STREQUAL "" OR
   NOT IS_DIRECTORY "${MEMFABRIC_HYBRID_HOME_PATH}")
  message(FATAL_ERROR
    "MEMFABRIC_HYBRID_HOME_PATH must point to the installed wgm-dev-310p "
    "MemFabric public SDK/runtime package")
endif()

set(vendor_name "memfabric_mc2" CACHE STRING "CANN custom OPP vendor")
set(ASCEND_COMPUTE_UNIT "ascend310p" CACHE STRING "CANN compute-unit family")
set(ENABLE_SOURCE_PACKAGE True CACHE BOOL "Package sources")
set(ENABLE_BINARY_PACKAGE True CACHE BOOL "Package binaries")

find_package(ASC REQUIRED HINTS
  "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")

set(package_name ${vendor_name})
npu_op_package(${package_name}
  TYPE RUN
  CONFIG
    ENABLE_SOURCE_PACKAGE ${ENABLE_SOURCE_PACKAGE}
    ENABLE_BINARY_PACKAGE ${ENABLE_BINARY_PACKAGE}
    INSTALL_PATH ${CMAKE_BINARY_DIR}/)

set(_mf_arch_dir "${CMAKE_SYSTEM_PROCESSOR}-linux")
find_path(MFMC2_MEMFABRIC_HOST_INCLUDE
  NAMES smem_shm.h
  PATHS
    "${MEMFABRIC_HYBRID_HOME_PATH}/include/smem/host"
    "${MEMFABRIC_HYBRID_HOME_PATH}/${_mf_arch_dir}/include/smem/host"
  NO_DEFAULT_PATH)
find_path(MFMC2_MEMFABRIC_DEVICE_INCLUDE
  NAMES smem_shm_aicore_base_sdma.h
  PATHS
    "${MEMFABRIC_HYBRID_HOME_PATH}/include/smem/device"
    "${MEMFABRIC_HYBRID_HOME_PATH}/${_mf_arch_dir}/include/smem/device"
  NO_DEFAULT_PATH)
find_library(MFMC2_MEMFABRIC_LIBRARY
  NAMES mf_smem
  PATHS
    "${MEMFABRIC_HYBRID_HOME_PATH}/${_mf_arch_dir}/lib64"
    "${MEMFABRIC_HYBRID_HOME_PATH}/${_mf_arch_dir}/lib"
  NO_DEFAULT_PATH)
if(NOT MFMC2_MEMFABRIC_LIBRARY)
  file(GLOB _mf_public_lib_dirs LIST_DIRECTORIES true
    "${MEMFABRIC_HYBRID_HOME_PATH}/*/lib64"
    "${MEMFABRIC_HYBRID_HOME_PATH}/*/lib")
  find_library(MFMC2_MEMFABRIC_LIBRARY
    NAMES mf_smem PATHS ${_mf_public_lib_dirs} NO_DEFAULT_PATH)
endif()
if(NOT MFMC2_MEMFABRIC_HOST_INCLUDE OR
   NOT MFMC2_MEMFABRIC_DEVICE_INCLUDE OR
   NOT MFMC2_MEMFABRIC_LIBRARY)
  message(FATAL_ERROR
    "MemFabric public contract incomplete: require smem_shm.h, "
    "smem_shm_aicore_base_sdma.h and libmf_smem.so")
endif()

find_program(MFMC2_BISHENG
  NAMES bisheng
  HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/bin"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/${_mf_arch_dir}/bin"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/bin"
  NO_DEFAULT_PATH)
if(NOT MFMC2_BISHENG)
  message(FATAL_ERROR "bisheng was not found under ASCEND_CANN_PACKAGE_PATH")
endif()

set(_cann_arch_root "${ASCEND_CANN_PACKAGE_PATH}/${_mf_arch_dir}")
if(NOT IS_DIRECTORY "${_cann_arch_root}")
  set(_cann_arch_root "${ASCEND_CANN_PACKAGE_PATH}")
endif()
set(MFMC2_DEVICE_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/libmfmc2_device.so")
add_custom_command(
  OUTPUT "${MFMC2_DEVICE_LIBRARY}"
  COMMAND "${MFMC2_BISHENG}"
          --npu-arch=dav-2002 -O2 -std=c++17 -w -shared -fPIC
          -x asc "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/memfabric310p_device.asc"
          -x none
          -o "${MFMC2_DEVICE_LIBRARY}"
          -I"${_cann_arch_root}/include"
          -I"${_cann_arch_root}/ascendc/include"
          -I"${ASCEND_CANN_PACKAGE_PATH}/include"
          -I"${MFMC2_MEMFABRIC_DEVICE_INCLUDE}"
          -L"${_cann_arch_root}/lib64"
          -L"${ASCEND_CANN_PACKAGE_PATH}/lib64"
          -lascendcl -lruntime -ldl -pthread -lm
          -Wl,-soname,libmfmc2_device.so
          -Wl,-rpath,"${_cann_arch_root}/lib64"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/memfabric310p_device.asc"
  COMMENT "Building dav-2002 MemFabric MC2 device pipeline"
  VERBATIM)
add_custom_target(mfmc2_device ALL DEPENDS "${MFMC2_DEVICE_LIBRARY}")
set_source_files_properties("${MFMC2_DEVICE_LIBRARY}" PROPERTIES GENERATED TRUE)

npu_op_library(cust_opapi ACLNN
  "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_mem_fabric_matmul_all_reduce.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime/memfabric_runtime.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime/memfabric310p_adapter.cpp")
add_dependencies(cust_opapi mfmc2_device)
target_include_directories(cust_opapi PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
  "${ASCEND_CANN_PACKAGE_PATH}/include"
  "${_cann_arch_root}/include"
  "${MFMC2_MEMFABRIC_HOST_INCLUDE}")
target_link_directories(cust_opapi PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${ASCEND_CANN_PACKAGE_PATH}/lib64"
  "${_cann_arch_root}/lib64")
get_filename_component(MFMC2_MEMFABRIC_LIBDIR "${MFMC2_MEMFABRIC_LIBRARY}" DIRECTORY)
target_link_directories(cust_opapi PRIVATE "${MFMC2_MEMFABRIC_LIBDIR}")
target_link_libraries(cust_opapi PRIVATE
  mfmc2_device
  mf_smem
  ascendcl dl pthread)
set_target_properties(cust_opapi PROPERTIES
  BUILD_RPATH "$ORIGIN;${MFMC2_MEMFABRIC_LIBDIR}"
  INSTALL_RPATH "$ORIGIN")

# CANN places custom operator API libraries/headers in
# vendors/<vendor>/op_api/{lib,include}.
npu_op_package_add(${package_name} LIBRARY cust_opapi)
npu_op_package_add(${package_name}
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/aclnn_mem_fabric_matmul_all_reduce.h"
  TYPE ACLNN)
npu_op_package_add(${package_name}
  FILES "${MFMC2_DEVICE_LIBRARY}"
  PACKAGE_PATH "op_api/lib")
# Bundle the public MemFabric host runtime used by libcust_opapi.so so vLLM
# does not need a separate LD_LIBRARY_PATH entry for libmf_smem.so. Any
# MemFabric platform/AICPU prerequisites remain a build/install prerequisite
# and are verified by tests/st/check_install.py on the target machine.
npu_op_package_add(${package_name}
  FILES "${MFMC2_MEMFABRIC_LIBRARY}"
  PACKAGE_PATH "op_api/lib")

message(STATUS "MemFabric MC2: CANN=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "MemFabric MC2: SDK=${MEMFABRIC_HYBRID_HOME_PATH}")
message(STATUS "MemFabric MC2: public host=${MFMC2_MEMFABRIC_HOST_INCLUDE}")
message(STATUS "MemFabric MC2: public device=${MFMC2_MEMFABRIC_DEVICE_INCLUDE}")
message(STATUS "MemFabric MC2: libmf_smem=${MFMC2_MEMFABRIC_LIBRARY}")
message(STATUS "MemFabric MC2: target=310P3/dav-2002 vendor=${vendor_name}")
