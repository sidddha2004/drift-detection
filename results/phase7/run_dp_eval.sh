PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 DRIFT_BENCH=v2
POL="ph_adapter_fed_dp_z0.5 ph_adapter_fed_dp_z1 ph_adapter_fed_dp_z2 ph_adapter_fed_dp_z4 ph_full_fed_E5_dp_z0.5 ph_full_fed_E5_dp_z1 ph_full_fed_E5_dp_z2 ph_full_fed_E5_dp_z4"
VAR="S1:virtual_gain_partial S1:virtual_rotation_partial S1:virtual_offset_partial S1:real_partial S2:virtual_rotation_partial S2:real_partial S3:virtual_rotation_partial S3:real_partial"
i=0
for SEEDS in "42 123 999" "2024 7 11" "13 17" "19 23"; do
  i=$((i+1))
  $PY experiments/phase4_streaming.py --confirmatory --tag dp_eval_$i --seeds $SEEDS --variants $VAR --policies $POL > results/phase7/dp_eval_$i.log 2>&1 &
done
wait
ls results/phase4_v2/dp_eval_*.json
