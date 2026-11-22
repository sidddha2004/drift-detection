PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 DRIFT_BENCH=v2
VAR="S1:virtual_gain_partial S1:virtual_rotation_partial S1:virtual_offset_partial S1:real_partial S2:virtual_rotation_partial S2:real_partial S3:virtual_rotation_partial S3:real_partial"
for s in 1001 1002 1003; do
  $PY experiments/phase4_streaming.py --tag dp_tune_$s --seeds $s --variants $VAR --policies ph_adapter_fed_dplog ph_full_fed_E5_dplog > results/phase7/dp_tune_$s.log 2>&1 &
done
wait
