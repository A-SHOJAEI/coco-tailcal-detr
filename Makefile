PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash

CONFIG ?= configs/smoke.yaml
PY := .venv/bin/python

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@bash scripts/bootstrap_venv.sh

data: setup
	@$(PY) -m tailcal_detr.pipeline data --config $(CONFIG)

train: setup
	@$(PY) -m tailcal_detr.pipeline train --config $(CONFIG)

eval: setup
	@$(PY) -m tailcal_detr.pipeline eval --config $(CONFIG)

report: setup
	@$(PY) -m tailcal_detr.pipeline report --config $(CONFIG)

all: data train eval report

clean:
	@rm -rf .venv artifacts runs

