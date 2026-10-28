PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 DRIFT_BENCH=v2
POL="frozen_E3b_personal periodic_E3c_full_fed adwin_full_fed_E4 ph_full_fed_E5 ph_align ph_head_long typed_learned ph_mode_oracle oracle_align typed_oracle"
SEEDS="1001 1002 1003"
$PY experiments/phase4_streaming.py --tag v2_A --seeds $SEEDS --variants S1:virtual_gain_partial S1:virtual_rotation_partial S1:virtual_offset_partial --policies $POL > results/phase4_v2/v2_A.log 2>&1 &
$PY experiments/phase4_streaming.py --tag v2_B --seeds $SEEDS --variants S1:real_partial S2:virtual_rotation_partial S2:real_partial --policies $POL > results/phase4_v2/v2_B.log 2>&1 &
$PY experiments/phase4_streaming.py --tag v2_C --seeds $SEEDS --variants S3:virtual_rotation_partial S3:real_partial S4:chest_rotation_partial S4:ankle_gain_partial --policies $POL > results/phase4_v2/v2_C.log 2>&1 &
wait
if [ -f results/phase4_v2/v2_A.json ] && [ -f results/phase4_v2/v2_B.json ] && [ -f results/phase4_v2/v2_C.json ]; then echo done > results/phase4_v2/V2_DONE; fi
