# Cross-platform shell configuration
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set shell := ["sh", "-c"]
set dotenv-load := true
set dotenv-required := false

[doc("All command information")]
default:
  @just --list --unsorted --list-heading $'AG2 commands\n'

# Tests

_beta_llm_filter := "not (openai or openai_realtime or gemini or gemini_realtime or anthropic or deepseek or ollama or bedrock or cerebras)"

[doc("Run beta tests")]
[group("tests")]
test-beta *params:
  pytest -vv --durations=10 --durations-min=1.0 \
    -m "{{ _beta_llm_filter }}" \
    test/beta/ {{ params }}

[doc("Run beta tests with coverage")]
[group("tests")]
test-beta-cov *params:
  pytest -vv --durations=10 --durations-min=1.0 \
    --cov=autogen/beta --cov-branch --cov-report=xml \
    -m "{{ _beta_llm_filter }}" \
    test/beta/ {{ params }}
  coverage report -m --include="autogen/beta/*"

_beta_llm_default_mark := "openai or gemini or anthropic or ollama or dashscope"

[doc("Run beta tests with LLM (e.g. just test-beta-llm openai)")]
[group("tests")]
test-beta-llm *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    -m "{{ _beta_llm_default_mark }}" \
    test/beta/ {{ params }}

[doc("Run beta tests with LLM and coverage")]
[group("tests")]
test-beta-llm-cov *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    --cov=autogen/beta/config --cov-branch --cov-report=xml \
    -m "{{ _beta_llm_default_mark }}" \
    test/beta/ {{ params }}
  coverage report -m --include="autogen/beta/config/*"

[doc("Run all beta tests (with and without LLMs)")]
[group("tests")]
test-beta-all *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    test/beta/ {{ params }}

[doc("Run all beta tests with coverage")]
[group("tests")]
test-beta-all-cov *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    --cov=autogen/beta --cov-branch --cov-report=xml \
    test/beta/ {{ params }}
  coverage report -m --include="autogen/beta/*"


# Linter

[doc("Ruff check")]
[group("linter")]
ruff-check *params:
  ruff check {{ params }}

[doc("Ruff format")]
[group("linter")]
ruff-format *params:
  ruff format {{ params }}

[doc("Check typos (codespell + pre-commit typos)")]
[group("linter")]
typos:
  pre-commit run --all-files codespell
  pre-commit run --all-files typos

[doc("Run ruff check + format")]
[group("linter")]
lint: ruff-check ruff-format typos


# Static analysis

[doc("Run mypy type check")]
[group("static analysis")]
mypy *params:
  mypy {{ params }}


# Pre-commit

[doc("Install pre-commit hooks")]
[group("pre-commit")]
pre-commit-install:
  pre-commit install

[doc("Run pre-commit on modified files")]
[group("pre-commit")]
pre-commit:
  pre-commit run

[doc("Run pre-commit on all files")]
[group("pre-commit")]
pre-commit-all:
  pre-commit run --all-files


# Docs

[doc("Build documentation")]
[group("docs")]
docs-build *params:
  cd website/mkdocs && python docs.py build {{ params }}

[doc("Serve documentation locally")]
[group("docs")]
docs-serve *params: docs-build
  cd website/mkdocs && python docs.py live {{ params }}
