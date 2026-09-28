from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

EXPECTED_ABI = 1


def candidate_op_libs() -> list[Path]:
    candidates: list[Path] = []
    for item in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if not item:
            continue
        root = Path(item)
        candidates.extend(
            [
                root / "op_api" / "lib" / "libcust_opapi.so",
                root / "libcust_opapi.so",
                root
                / "vendors"
                / "memfabric_mc2"
                / "op_api"
                / "lib"
                / "libcust_opapi.so",
            ]
        )
    opp = os.environ.get("ASCEND_OPP_PATH")
    if opp:
        candidates.append(
            Path(opp)
            / "vendors"
            / "memfabric_mc2"
            / "op_api"
            / "lib"
            / "libcust_opapi.so"
        )
    seen: set[Path] = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]


def find_op_lib() -> Path:
    for candidate in candidate_op_libs():
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "memfabric_mc2 custom OPP not found. Install custom_opp_*.run and "
        "source vendors/memfabric_mc2/bin/set_env.bash before testing."
    )


def require_ldd_clean(path: Path) -> None:
    proc = subprocess.run(
        ["ldd", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ldd failed for {path}:\n{proc.stdout}")
    if "not found" in proc.stdout:
        raise RuntimeError(f"unresolved runtime dependency for {path}:\n{proc.stdout}")
    print(proc.stdout.rstrip())


def require_relocatable_needed(path: Path) -> None:
    proc = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"readelf failed for {path}:\n{proc.stdout}")
    for line in proc.stdout.splitlines():
        if "(NEEDED)" not in line:
            continue
        left = line.find("[")
        right = line.find("]", left + 1)
        if left < 0 or right < 0:
            continue
        needed = line[left + 1 : right]
        if needed.startswith("/"):
            raise RuntimeError(
                f"non-relocatable DT_NEEDED in {path}: {needed}. "
                "The installed OPP must not depend on a build-machine path."
            )


def main() -> None:
    op_lib = find_op_lib()
    lib_dir = op_lib.parent
    mf_lib = lib_dir / "libmf_smem.so"
    device_lib = lib_dir / "libmfmc2_device.so"
    if not mf_lib.is_file():
        raise RuntimeError(f"bundled public MemFabric runtime missing: {mf_lib}")
    if not device_lib.is_file():
        raise RuntimeError(f"packaged device pipeline missing: {device_lib}")

    for path in (device_lib, mf_lib, op_lib):
        require_relocatable_needed(path)
        require_ldd_clean(path)

    lib = ctypes.CDLL(str(op_lib), mode=ctypes.RTLD_GLOBAL)
    required = [
        "aclnnMemFabricMatmulAllReduceGetWorkspaceSize",
        "aclnnMemFabricMatmulAllReduce",
        "mfmc2RuntimeAbiVersion",
        "mfmc2RuntimeShutdown",
        "mfmc2RuntimeGetLastError",
        "mfmc2RuntimeDebugSnapshot",
    ]
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(f"custom OPP is missing public symbols: {missing}")

    abi = lib.mfmc2RuntimeAbiVersion
    abi.restype = ctypes.c_uint32
    actual = int(abi())
    if actual != EXPECTED_ABI:
        raise RuntimeError(f"runtime ABI mismatch: expected={EXPECTED_ABI}, got={actual}")

    print(f"PASS op_lib={op_lib}")
    print(f"PASS runtime_abi={actual}")
    print("PASS all direct ELF dependencies resolved")
    print("PASS no absolute DT_NEEDED entries")
    print(
        "NOTE: this checks shared-library closure. The following real-hardware "
        "functional ST still verifies MemFabric 310P AICPU/orchestrator deployment."
    )


if __name__ == "__main__":
    main()
