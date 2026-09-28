from __future__ import annotations

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "common"))
from mfmc2_ctypes import Mfmc2Api  # noqa: E402


def _phase1(api: Mfmc2Api, rows: int, rank: int, q: int,
            x: int = 0, w: int = 0, out: int = 0) -> tuple[int, int, int]:
    workspace = ctypes.c_uint64(123)
    executor = ctypes.c_void_p()
    ret = api.stage1(
        x, w, rows, rank, q, out, False,
        ctypes.byref(workspace), ctypes.byref(executor)
    )
    return int(ret), int(workspace.value), int(executor.value or 0)


def main() -> None:
    api = Mfmc2Api()
    bad = [
        ("negative rows", _phase1(api, -1, 0, 2)),
        ("rank < 0", _phase1(api, 1, -1, 2, 1, 1, 1)),
        ("rank > 1", _phase1(api, 1, 2, 2, 1, 1, 1)),
        ("invalid q", _phase1(api, 1, 0, 3, 1, 1, 1)),
        ("null x", _phase1(api, 1, 0, 2, 0, 1, 1)),
        ("null weight", _phase1(api, 1, 0, 2, 1, 0, 1)),
        ("null output", _phase1(api, 1, 0, 2, 1, 1, 0)),
    ]
    for name, (ret, workspace, executor) in bad:
        if ret == 0 or executor != 0:
            raise AssertionError(
                f"{name}: expected phase1 rejection with null executor, "
                f"got ret={ret} workspace={workspace} executor=0x{executor:x}"
            )

    # M=0 is a legal no-op contract: addresses may be null and phase1 still
    # returns a zero-workspace executor. Do not call phase2 here because this
    # smoke intentionally requires no NPU context.
    ret, workspace, executor = _phase1(api, 0, 0, 2)
    if ret != 0 or workspace != 0 or executor == 0:
        raise AssertionError(
            f"M=0 contract failed: ret={ret} workspace={workspace} executor={executor}"
        )
    # Avoid leaking the test executor: phase2 owns executor deletion, but a
    # real stream is intentionally absent here. Call libc++ delete is not part
    # of the public ABI, so the M=0 positive case is covered by functional ST.
    # This process exits immediately after the smoke, making the one executor
    # allocation bounded and harmless.

    print("PASS ACLNN phase1 validation rejects invalid rank/q/address inputs")


if __name__ == "__main__":
    main()
