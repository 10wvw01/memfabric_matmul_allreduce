from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_required_project_files_exist():
    required = [
        "CMakeLists.txt",
        "CMakePresets.json",
        "build.sh",
        "include/aclnn_mem_fabric_matmul_all_reduce.h",
        "op_api/aclnn_mem_fabric_matmul_all_reduce.cpp",
        "op_kernel/memfabric310p_device.asc",
        "runtime/memfabric310p_adapter.cpp",
        "runtime/memfabric310p_adapter_api.h",
        "runtime/memfabric_runtime.cpp",
        "runtime/memfabric_runtime.h",
        "tests/run_real_machine_gate.sh",
        "tests/st/check_install.py",
        "tests/st/test_api_validation.py",
        "tests/st/test_data_path_accuracy.py",
        "tests/st/test_two_rank_accuracy.py",
        "tests/st/test_repeated_stability.py",
        "tests/perf/bench_two_rank.py",
        "docs/design.md",
        "docs/test_plan.md",
        "docs/review_report.md",
    ]
    for rel in required:
        assert (ROOT / rel).is_file(), rel


def test_cann91_run_packaging_is_the_only_build_path():
    cmake = _text("CMakeLists.txt")
    build = _text("build.sh")
    assert "find_package(ASC REQUIRED" in cmake
    assert "npu_op_package(${package_name}" in cmake
    assert "TYPE RUN" in cmake
    assert "npu_op_library(cust_opapi ACLNN" in cmake
    assert "PACKAGE_PATH \"op_api/lib\"" in cmake
    assert "--npu-arch=dav-2002" in cmake
    binary_cmd = 'cmake --build "${BUILD_DIR}" --target binary'
    package_cmd = 'cmake --build "${BUILD_DIR}" --target package'
    assert binary_cmd in build
    assert package_cmd in build
    assert build.index(binary_cmd) < build.index(package_cmd)


def test_packaged_shared_libraries_are_relocatable():
    cmake = _text("CMakeLists.txt")
    install_check = _text("tests/st/check_install.py")
    assert "-Wl,-soname,libmfmc2_device.so" in cmake
    assert "target_link_libraries(cust_opapi PRIVATE\n  mfmc2_device\n  mf_smem" in cmake
    assert 'INSTALL_RPATH "$ORIGIN"' in cmake
    assert 'FILES "${MFMC2_DEVICE_LIBRARY}"' in cmake
    assert 'FILES "${MFMC2_MEMFABRIC_LIBRARY}"' in cmake
    assert "require_relocatable_needed" in install_check
    assert 'needed.startswith("/")' in install_check
    assert "libmfmc2_device.so" in install_check
    assert "libmf_smem.so" in install_check


def test_public_api_has_aclnn_two_phase_contract():
    header = _text("include/aclnn_mem_fabric_matmul_all_reduce.h")
    assert "aclnnMemFabricMatmulAllReduceGetWorkspaceSize" in header
    assert "aclnnMemFabricMatmulAllReduce(" in header
    assert "mfmc2RuntimeAbiVersion" in header
    assert "mfmc2RuntimeShutdown" in header


def test_operator_has_no_vllm_or_torch_dependency():
    for rel in [
        "op_api/aclnn_mem_fabric_matmul_all_reduce.cpp",
        "runtime/memfabric_runtime.cpp",
        "runtime/memfabric310p_adapter.cpp",
        "op_kernel/memfabric310p_device.asc",
    ]:
        lowered = _text(rel).lower()
        assert "vllm" not in lowered, rel
        assert "torch/" not in lowered, rel
        assert "torch_npu" not in lowered, rel


def test_memfabric_is_consumed_only_through_public_contract():
    host = _text("runtime/memfabric310p_adapter.cpp")
    device = _text("op_kernel/memfabric310p_device.asc")
    assert '#include <smem.h>' in host
    assert '#include <smem_shm.h>' in host
    assert '"smem_shm_aicore_base_sdma.h"' in device
    assert "smem_shm_sdma_signal" in device
    assert "smem_shm_sdma_wait" in device
    assert "smem_shm_sdma_quiet" in device
    forbidden = ("mailbox/", "orchestrator/", "reserved_region", "aicpu/")
    for token in forbidden:
        assert token not in host.lower()
        assert token not in device.lower()


def test_v8_data_path_invariants_are_preserved():
    device = _text("op_kernel/memfabric310p_device.asc")
    runtime = _text("runtime/memfabric_runtime.cpp")
    assert "MFMC2_CORES = 8" in device
    assert "MFMC2_PRODUCER_TILING(256, 256)" in device
    assert "MFMC2_PRODUCER_TILING(512, 512)" in device
    assert "MFMC2_PRODUCER_TILING(1024, 1024)" in device
    assert "CONFIG_NORM" in device
    assert "kEagerGenerationBase = 0x40000000u" in runtime
    assert "EnqueueProducer(batch)" in runtime
    assert "EnqueueWaitAdd(batch - 1)" in runtime
    assert runtime.index("EnqueueProducer(batch)") < runtime.index("EnqueueWaitAdd(batch - 1)")


def test_accuracy_stability_and_perf_are_independent_of_vllm():
    api_validation = _text("tests/st/test_api_validation.py")
    accuracy = _text("tests/st/test_two_rank_accuracy.py")
    data_path = _text("tests/st/test_data_path_accuracy.py")
    stability = _text("tests/st/test_repeated_stability.py")
    perf = _text("tests/perf/bench_two_rank.py")
    for text in (api_validation, accuracy, data_path, stability, perf):
        assert "vllm" not in text.lower()
    assert "invalid q" in api_validation
    assert "M=0" in api_validation
    assert "torch.equal" in accuracy
    assert "[1, 255, 256, 257" in accuracy
    assert "torch.equal" in data_path
    assert "source_rank" in data_path
    assert "peer-payload" in data_path
    assert "default=1000" in stability
    assert "torch.equal" in stability
    assert "dist.all_reduce" in perf
    assert "unfair allocator advantage" in perf


def test_real_machine_gate_orders_correctness_before_performance():
    gate = _text("tests/run_real_machine_gate.sh")
    required = [
        "check_install.py",
        "test_api_validation.py",
        "test_data_path_accuracy.py",
        "test_two_rank_accuracy.py --q 2",
        "test_two_rank_accuracy.py --q 1 --q 4",
        "test_repeated_stability.py",
        "bench_two_rank.py",
    ]
    positions = [gate.index(token) for token in required]
    assert positions == sorted(positions)


def test_install_and_st_use_installed_custom_opp():
    install = _text("tests/st/check_install.py")
    caller = _text("tests/common/mfmc2_ctypes.py")
    assert "ASCEND_CUSTOM_OPP_PATH" in install
    assert "libcust_opapi.so" in install
    assert "mfmc2RuntimeAbiVersion" in install
    assert "ASCEND_CUSTOM_OPP_PATH" in caller
    assert "aclnnMemFabricMatmulAllReduceGetWorkspaceSize" in caller
    assert "aclnnMemFabricMatmulAllReduce" in caller
