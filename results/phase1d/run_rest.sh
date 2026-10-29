PY=../venv/Scripts/python.exe
C=configs/phase1_v2_trimmed.yaml
export PYTHONIOENCODING=utf-8
$PY experiments/phase1c_fedavg_ft.py --config $C --split-mode blocked --out results/phase1d/fedavg_ft_blocked.json > results/phase1d/fedavg_ft_blocked.log 2>&1 || exit 1
$PY experiments/phase1_baselines.py --config $C --split-mode temporal --out results/phase1d/baselines_temporal.json > results/phase1d/baselines_temporal.log 2>&1 || exit 1
$PY experiments/phase1c_fedavg_ft.py --config $C --split-mode temporal --out results/phase1d/fedavg_ft_temporal.json > results/phase1d/fedavg_ft_temporal.log 2>&1 || exit 1
echo done > results/phase1d/DONE
