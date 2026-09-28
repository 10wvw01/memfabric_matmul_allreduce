#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${ASCEND_RT_VISIBLE_DEVICES:=0,1}"
: "${MFMC2_STORE_URL:=tcp://127.0.0.1:8581}"
: "${MFMC2_LOCAL_BYTES:=$((96 * 1024 * 1024))}"
: "${MFMC2_GATE_WARMUP:=20}"
: "${MFMC2_GATE_ITERS:=100}"
: "${MFMC2_GATE_STABILITY_ITERS:=1000}"

export ASCEND_RT_VISIBLE_DEVICES MFMC2_STORE_URL MFMC2_LOCAL_BYTES

echo "[1/8] static source contracts"
python3 -m pytest tests/ut -q

echo "[2/8] installed OPP / ELF / ABI"
python3 tests/st/check_install.py
python3 tests/st/test_api_validation.py

echo "[3/8] isolated local-MM / peer-SDMA payload correctness"
python3 tests/st/test_data_path_accuracy.py

echo "[4/8] full TP=2 correctness q=2"
python3 tests/st/test_two_rank_accuracy.py --q 2

echo "[5/8] full TP=2 correctness q=1/q=4"
python3 tests/st/test_two_rank_accuracy.py --q 1 --q 4

echo "[6/8] repeated mixed-M protocol stability"
python3 tests/st/test_repeated_stability.py \
  --q 2 --iterations "${MFMC2_GATE_STABILITY_ITERS}"

echo "[7/8] standalone stock-vs-fused performance"
python3 tests/perf/bench_two_rank.py \
  --q 2 --warmup "${MFMC2_GATE_WARMUP}" --iters "${MFMC2_GATE_ITERS}"

echo "[8/8] gate complete"
echo "PASS: standalone pre-vLLM gate completed."
echo "NEXT: capture CANN profiler overlap, then execute vllm-ascend eager/ACL-Graph/10min gates."
