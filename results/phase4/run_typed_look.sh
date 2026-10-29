PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
while [ ! -f results/phase4/FIRST_LOOK_DONE ]; do sleep 20; done
$PY experiments/phase4_streaming.py --tag typed_virtual --seeds 1001 1002 1003 \
  --variants S2:virtual_offset_partial S2:virtual_rotation_partial S3:virtual_offset_partial S3:virtual_rotation_partial S4:hand_offset_partial S4:ankle_gain_partial S4:chest_rotation_partial \
  --policies frozen_E3b_personal typed_oracle fixed_align_oneshot fixed_align_track fixed_full_fed fixed_adapter_fed > results/phase4/typed_virtual.log 2>&1
$PY experiments/phase4_streaming.py --tag typed_real --seeds 1001 1002 1003 \
  --variants S2:real_partial S3:real_partial \
  --policies frozen_E3b_personal typed_oracle fixed_align_oneshot fixed_head_long fixed_full_fed > results/phase4/typed_real.log 2>&1
echo done > results/phase4/TYPED_LOOK_DONE
