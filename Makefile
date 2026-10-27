PY := .venv/bin/python

.PHONY: all setup run test clean

all: setup run test

setup: .venv/bin/activate
.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt
	touch .venv/bin/activate

run: setup
	$(PY) run.py

test: setup
	$(PY) -m pytest -q

clean:
	rm -rf results/* figures/*
