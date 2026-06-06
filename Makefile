.PHONY: test smoke forge-zero benchmark tune-safety scale-study reference data train evaluate dashboard demo

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

smoke:
	PYTHONPATH=. python -m ghostfighter.cli all --out runs/smoke --episodes-per-style 2 --epochs 1 --eval-episodes 8 --max-steps 40

forge-zero:
	PYTHONPATH=. python -m ghostfighter.cli forge-zero --out runs/default/data/traces.npz --episodes-per-style 80 --variants-per-archetype 8

benchmark:
	PYTHONPATH=. python -m ghostfighter.cli benchmark --model runs/default/models/ghost_policy.pt --out runs/default/reports --suite all --episodes 80

tune-safety:
	PYTHONPATH=. python -m ghostfighter.cli tune-safety --model runs/default/models/ghost_policy.pt --out runs/default/reports --suite regression --episodes 20

scale-study:
	PYTHONPATH=. python -m ghostfighter.cli scale-study --out runs/default/scaling --episodes-schedule 8,16,32 --epochs 3 --eval-episodes 24

reference:
	PYTHONPATH=. python -m ghostfighter.cli all --out runs/reference --episodes-per-style 60 --epochs 6 --eval-episodes 32 --max-steps 90 --stress

data:
	PYTHONPATH=. python -m ghostfighter.cli generate-data --source attributes --out runs/default/data/traces.npz --episodes-per-style 80 --variants-per-archetype 8

train:
	PYTHONPATH=. python -m ghostfighter.cli train --data runs/default/data/traces.npz --out runs/default/models --epochs 8 --batch-size 2048

evaluate:
	PYTHONPATH=. python -m ghostfighter.cli evaluate --model runs/default/models/ghost_policy.pt --out runs/default/reports --episodes 80 --scripted-baseline --stress

dashboard:
	PYTHONPATH=. python -m ghostfighter.cli dashboard --reports runs/default/reports

demo:
	PYTHONPATH=. python -m ghostfighter.cli demo --model runs/default/models/ghost_policy.pt --out runs/default/videos/ghostfighter_demo.gif --style pressure
