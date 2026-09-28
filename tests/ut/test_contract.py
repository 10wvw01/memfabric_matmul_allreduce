from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_required_project_files_exist():
    required = [
        "CMakeLists.txt",
        "op_host/memfabric_matmul_allreduce_def.cpp",
        "op_host/memfabric_matmul_allreduce_tiling.cpp",
        "op_kernel/memfabric_matmul_allreduce.cpp",
        "runtime/memfabric_runtime.cpp",
        "docs/design.md",
        "docs/test_plan.md",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel


def test_v1_contract_is_model_agnostic():
    text = (ROOT / "include/memfabric_matmul_allreduce_contract.h").read_text()
    lowered = text.lower()
    assert "qwen" not in lowered
    assert "o_proj" not in lowered
    assert "kLocalK = 2048" in text
    assert "kN = 2048" in text
    assert "kWorldSize = 2" in text
