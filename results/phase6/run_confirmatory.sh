PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 DRIFT_BENCH=v2
POL="frozen_E3_global frozen_E3b_personal periodic_E3c_full_fed adwin_full_fed_E4 ph_full_fed_E5 ph_fedprox_E7a ph_adapter_fed ph_full_local ph_align ph_head_long typed_learned ph_mode_oracle typed_oracle mode_typed_oracle"
VAR="S1:virtual_gain_partial S1:virtual_rotation_partial S1:virtual_offset_partial S1:real_partial S2:virtual_rotation_partial S2:real_partial S3:virtual_rotation_partial S3:real_partial S4:chest_rotation_partial S4:ankle_gain_partial"
$PY experiments/phase4_streaming.py --confirmatory --tag conf_1 --seeds 42 123 999 2024 7 --variants $VAR --policies $POL > results/phase6/conf_1.log 2>&1 &
$PY experiments/phase4_streaming.py --confirmatory --tag conf_2 --seeds 11 13 17 19 23 --variants $VAR --policies $POL > results/phase6/conf_2.log 2>&1 &
wait
if [ -f results/phase4_v2/conf_1.json ] && [ -f results/phase4_v2/conf_2.json ]; then echo done > results/phase6/CONF_DONE; fi
