.PHONY: install test compile privacy-audit

install:
	python -m pip install -e '.[test]'

test:
	pytest -q

compile:
	python -m compileall -q src tests

privacy-audit:
	! rg -n -i '(BEGIN .*PRIVATE KEY|api[_-]?key|access[_-]?token|password|passwd|/Users/|/root/)' --glob '!*.pdf' --glob '!Makefile' .
