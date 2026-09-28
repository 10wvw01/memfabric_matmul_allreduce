from __future__ import annotations

import ctypes
import os
from pathlib import Path


class Mfmc2Api:
    """Minimal standalone caller for the installed custom OPP.

    This deliberately does not import vllm-ascend. PyTorch/torch_npu may be
    used by tests only to allocate/prepare NPU tensors and build references.
    """

    def __init__(self) -> None:
        self.op_lib = ctypes.CDLL(str(self._find_op_lib()), mode=ctypes.RTLD_GLOBAL)
        self.acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)

        self.stage1 = self.op_lib.aclnnMemFabricMatmulAllReduceGetWorkspaceSize
        self.stage1.restype = ctypes.c_int
        self.stage1.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_uint64,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.stage2 = self.op_lib.aclnnMemFabricMatmulAllReduce
        self.stage2.restype = ctypes.c_int
        self.stage2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.shutdown_fn = self.op_lib.mfmc2RuntimeShutdown
        self.shutdown_fn.restype = ctypes.c_int
        self.last_error_fn = self.op_lib.mfmc2RuntimeGetLastError
        self.last_error_fn.restype = ctypes.c_char_p

        self.acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.acl.aclrtCreateStream.restype = ctypes.c_int
        self.acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        self.acl.aclrtSynchronizeStream.restype = ctypes.c_int
        self.acl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
        self.acl.aclrtDestroyStream.restype = ctypes.c_int

    @staticmethod
    def _candidate_libs() -> list[Path]:
        candidates: list[Path] = []
        for item in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
            if not item:
                continue
            root = Path(item)
            # CANN 9.1 accepts either an installed vendor root or, for dynamic
            # operator libraries, a direct .../op_api/lib entry.
            candidates.extend(
                [
                    root / "op_api" / "lib" / "libcust_opapi.so",
                    root / "libcust_opapi.so",
                    root / "vendors" / "memfabric_mc2" / "op_api" / "lib" / "libcust_opapi.so",
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

    @classmethod
    def _find_op_lib(cls) -> Path:
        for candidate in cls._candidate_libs():
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "libcust_opapi.so for memfabric_mc2 was not found; install the run package "
            "and source its vendors/memfabric_mc2/bin/set_env.bash"
        )

    def last_error(self) -> str:
        raw = self.last_error_fn()
        return raw.decode("utf-8", errors="replace") if raw else ""

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        ret = self.acl.aclrtCreateStream(ctypes.byref(stream))
        if ret != 0 or not stream.value:
            raise RuntimeError(f"aclrtCreateStream failed: {ret}")
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream and stream.value:
            self.acl.aclrtDestroyStream(stream)

    def launch(self, x_addr: int, weight_addr: int, rows: int, rank: int,
               q: int, output_addr: int, stream: ctypes.c_void_p,
               graph_capturing: bool = False, synchronize: bool = True) -> None:
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = self.stage1(
            x_addr,
            weight_addr,
            rows,
            rank,
            q,
            output_addr,
            graph_capturing,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"ACLNN phase1 failed: ret={ret}, detail={self.last_error()}")
        if workspace_size.value != 0:
            raise RuntimeError(f"unexpected workspace_size={workspace_size.value}")
        ret = self.stage2(None, 0, executor, stream)
        if ret != 0:
            raise RuntimeError(f"ACLNN phase2 failed: ret={ret}, detail={self.last_error()}")
        if synchronize:
            ret = self.acl.aclrtSynchronizeStream(stream)
            if ret != 0:
                raise RuntimeError(f"aclrtSynchronizeStream failed: {ret}")

    def shutdown(self) -> None:
        ret = self.shutdown_fn()
        if ret != 0:
            raise RuntimeError(f"mfmc2RuntimeShutdown failed: ret={ret}, detail={self.last_error()}")
