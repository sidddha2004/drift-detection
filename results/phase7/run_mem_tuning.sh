PY=../venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 DRIFT_BENCH=v2
POL="typed_learned $($PY -c "import sys; sys.path.insert(0,'.'); from experiments.phase4_streaming import POLICIES; print(' '.join(p for p in POLICIES if p.startswith('typed_memory')))")"
VAR="S3:virtual_rotation_partial S3:real_partial"
for s in 1001 1002 1003; do
  $PY experiments/phase4_streaming.py --tag mem_tune_$s --seeds $s --variants $VAR --policies $POL > results/phase7/mem_tune_$s.log 2>&1 &
done
wait
