PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY experiments/phase4_streaming.py --tag s1_virtual_first_look --seeds 1001 1002 1003 \
  --variants S1:virtual_offset_partial S1:virtual_gain_partial S1:virtual_rotation_partial S1:virtual_offset_full \
  --policies frozen_E3_global frozen_E3b_personal periodic_E3c_full_fed oracle_full_fed oracle_adapter_fed oracle_align ph_full_fed_E5 ph_align ph_adapter_fed > results/phase4/s1_virtual_first_look.log 2>&1
$PY experiments/phase4_streaming.py --tag s1_real_first_look --seeds 1001 1002 1003 \
  --variants S1:real_partial \
  --policies frozen_E3_global frozen_E3b_personal periodic_E3c_full_fed oracle_full_fed oracle_head_long oracle_adapter_long oracle_full_long ph_full_fed_E5 ph_head_long > results/phase4/s1_real_first_look.log 2>&1
echo done > results/phase4/FIRST_LOOK_DONE
