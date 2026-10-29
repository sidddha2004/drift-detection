set -e
PY=../venv/Scripts/python.exe
C=configs/phase1_v2_trimmed.yaml
for m in blocked temporal; do
  $PY experiments/phase1_baselines.py --config $C --split-mode $m --out results/phase1d/baselines_$m.json > results/phase1d/baselines_$m.log 2>&1
  $PY experiments/phase1c_fedavg_ft.py --config $C --split-mode $m --out results/phase1d/fedavg_ft_$m.json > results/phase1d/fedavg_ft_$m.log 2>&1
done
echo done > results/phase1d/DONE
