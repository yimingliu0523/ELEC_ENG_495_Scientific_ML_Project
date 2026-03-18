PYTHON ?= python

.PHONY: data train eval figs reproduce test

data:
	$(PYTHON) scripts/generate_data.py --config configs/data_default.yaml

train:
	$(PYTHON) -m src.training.train_mlp --config configs/train_mlp.yaml
	$(PYTHON) -m src.training.train_resnet --config configs/train_resnet.yaml
	$(PYTHON) -m src.training.train_neural_ode --config configs/train_neural_ode.yaml

eval:
	$(PYTHON) -m src.evaluation.evaluate_model --model mlp --config configs/eval_default.yaml
	$(PYTHON) -m src.evaluation.evaluate_model --model resnet --config configs/eval_default.yaml
	$(PYTHON) -m src.evaluation.evaluate_model --model neural_ode --config configs/eval_default.yaml
	$(PYTHON) -m src.evaluation.compare_models --config configs/eval_default.yaml

figs:
	$(PYTHON) -m src.visualization.make_report_figures --config configs/eval_default.yaml

reproduce: data train eval figs

test:
	$(PYTHON) -m pytest -q
