from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

EXPECTED_ABI = 1


def candidate_vendor_roots() -> list[Path]:
    roots: list[Path] = []
    for item in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if item:
            p = Path(item)
            roots.extend([p, p / "vendors" / "memfabric_mc2"])
    opp = os.environ.get("ASCEND_OPP_PATH")
    if opp:
        roots.append(Path(opp) / "vendors" / "memfabric_mc2")
    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    return [p for p in roots if not (p in seen or seen.add(p))]


def find_vendor_root() -> Path:
    for root in candidate_vendor_roots():
        if (root / "op_api" / "lib" / "libcust_opapi.so").is_file():
            return root
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


def main() -> None:
    root = find_vendor_root()
    op_lib = root / "op_api" / "lib" / "libcust_opapi.so"
    mf_lib = root / "op_api" / "lib" / "libmf_smem.so"
    if not mf_lib.is_file():
        raise RuntimeError(f"bundled public MemFabric runtime missing: {mf_lib}")

    require_ldd_clean(mf_lib)
    require_ldd_clean(op_lib)

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

    print(f"PASS vendor_root={root}")
    print(f"PASS runtime_abi={actual}")
    print("PASS all direct ELF dependencies resolved")
    print(
        "NOTE: this checks shared-library closure. The following real-hardware "
        "functional ST still verifies MemFabric 310P AICPU/orchestrator deployment."
    )


if __name__ == "__main__":
    main()
