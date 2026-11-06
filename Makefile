PY ?= .venv/bin/python
OPTRISK ?= .venv/bin/optrisk

.PHONY: install test demo validate quantlib benchmark reproduce

install:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev,reference]"

test:
	$(PY) -m pytest

# The main demonstration (about a minute, writes reports/demo/)
demo:
	$(OPTRISK) demo

validate:
	$(OPTRISK) validate

quantlib:
	$(OPTRISK) quantlib-compare

benchmark:
	$(OPTRISK) benchmark

# Everything: demo + full validation + QuantLib comparison + benchmarks -> reports/
reproduce:
	$(OPTRISK) reproduce
