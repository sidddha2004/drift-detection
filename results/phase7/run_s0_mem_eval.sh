# Supervisor feedback C8/C9 (2026-11-24): S0 no-drift control and the A6-selected memory config on S3, 10 evaluation seeds.
PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 DRIFT_BENCH=v2
i=0
for SEEDS in "42 123 999" "2024 7 11" "13 17" "19 23"; do
  i=$((i+1))
  ( $PY experiments/phase4_streaming.py --confirmatory --tag s0_eval_$i --seeds $SEEDS --variants S0:none \
      --policies frozen_E3b_personal typed_learned ph_head_long ph_align ph_adapter_fed ph_full_fed_E5 periodic_E3c_full_fed > results/phase7/s0_eval_$i.log 2>&1
    $PY experiments/phase4_streaming.py --confirmatory --tag mem_eval_$i --seeds $SEEDS --variants S3:virtual_rotation_partial S3:real_partial \
      --policies typed_memory_K0_t02 typed_memory_K4_t02 > results/phase7/mem_eval_$i.log 2>&1 ) &
done
wait
ls results/phase4_v2/s0_eval_*.json results/phase4_v2/mem_eval_*.json
