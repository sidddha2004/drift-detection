PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY experiments/phase4_streaming.py --tag learned_s1 --seeds 1001 1002 1003 \
  --variants S1:virtual_offset_partial S1:virtual_gain_partial S1:virtual_rotation_partial S1:virtual_offset_full S1:real_partial \
  --policies typed_learned ph_mode_oracle ph_head_long > results/phase4/learned_s1.log 2>&1
$PY experiments/phase4_streaming.py --tag learned_s234 --seeds 1001 1002 1003 \
  --variants S2:virtual_offset_partial S2:virtual_rotation_partial S2:real_partial S3:virtual_offset_partial S3:virtual_rotation_partial S3:real_partial S4:hand_offset_partial S4:ankle_gain_partial S4:chest_rotation_partial \
  --policies frozen_E3b_personal typed_learned ph_mode_oracle ph_align ph_head_long ph_full_fed_E5 > results/phase4/learned_s234.log 2>&1
echo done > results/phase4/LEARNED_DONE
