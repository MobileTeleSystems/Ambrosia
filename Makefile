VENV=.venv

ifeq (${OS},Windows_NT)
	BIN=${VENV}/Scripts
else
	BIN=${VENV}/bin
endif

export PATH := $(BIN):$(PATH)

FLAKE=flake8
PYLINT=pylint
ISORT=isort
BLACK=black
PYTEST=pytest
COVERAGE=coverage

SOURCES=ambrosia
TESTS=tests
REPORTS=reports


# Installation

reports:
	@mkdir ${REPORTS}

.venv:
	@echo "Creating virtualenv...\t\t"
	poetry install --no-root
	@echo "[Installed]"

install: .venv reports


# Linters

.isort:
	@echo "Running isort checks..."
	@${ISORT} --check ${SOURCES} ${TESTS}
	@echo "[Isort checks finished]"

.black:
	@echo "Running black checks..."
	@${BLACK} --check --diff ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Black checks finished]"

.pylint: reports
	@echo "Running pylint checks..."
	@${PYLINT} ${SOURCES} ${TESTS} --exit-zero 
	@${PYLINT} ${SOURCES} ${TESTS} --exit-zero > ${REPORTS}/pylint.txt
	@echo "[Pylint checks finished]"

.flake8:
	@echo "Running flake8 checks...\t"
	@${FLAKE} ${SOURCES} ${TESTS} --exit-zero 
	@echo "[Flake8 checks finished]"


# Fixers & formatters

.isort_fix:
	@echo "Fixing isort..."
	@${ISORT} ${SOURCES} ${TESTS}
	@echo "[Isort fixed]"

.black_fix:
	@echo "Formatting with black..."
	@${BLACK} -q  ${SOURCES} ${TESTS}
	@echo "[Black fixed]"


# Tests

.pytest:
	@echo "Running pytest checks...\t"
	@PYTHONPATH=. ${PYTEST} --cov=${SOURCES} --cov-report=xml:${REPORTS}/coverage.xml

coverage: .venv reports
	@echo "Running coverage..."
	${COVERAGE} run --source ${SOURCES} --module pytest
	${COVERAGE} report
	${COVERAGE} html -d ${REPORTS}/coverage_html
	${COVERAGE} xml -o ${REPORTS}/coverage.xml -i


# Generalization

.autoformat: .isort_fix .black_fix
autoformat: .venv .autoformat

.lint: .isort .black .pylint .flake8
lint: .venv .lint

.test: .pytest 
test: .venv .test


# Cleaning

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf ${VENV}
	@rm -rf ${REPORTS}
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +

reinstall: clean install