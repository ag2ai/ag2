# Cross-platform shell configuration
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set shell := ["sh", "-c"]
set dotenv-load := true
set dotenv-required := false

[doc("All command information")]
default:
  @just --list --unsorted --list-heading $'AG2 commands\n'

# Tests

_llm_filter := "not (openai or openai_realtime or gemini or gemini_realtime or anthropic or zai or deepseek or ollama or bedrock or cerebras)"

[doc("Run tests")]
[group("tests")]
test *params:
  pytest -vv --durations=10 --durations-min=1.0 \
    -m "{{ _llm_filter }}" \
    test/ {{ params }}

[doc("Run tests with coverage")]
[group("tests")]
test-cov *params:
  pytest -vv --durations=10 --durations-min=1.0 \
    --cov=ag2 --cov-branch --cov-report=xml \
    -m "{{ _llm_filter }}" \
    test/ {{ params }}
  coverage report -m --include="ag2/*"

_llm_default_mark := "openai or gemini or anthropic or zai or ollama or dashscope"

[doc("Run tests with LLM (e.g. just test-llm openai)")]
[group("tests")]
test-llm mark=_llm_default_mark *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    -m "{{ mark }}" \
    test/ {{ params }}

[doc("Run tests with LLM and coverage (e.g. just test-llm-cov openai)")]
[group("tests")]
test-llm-cov mark=_llm_default_mark *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    --cov=ag2/config --cov-branch --cov-report=xml \
    -m "{{ mark }}" \
    test/ {{ params }}
  coverage report -m --include="ag2/config/*"

[doc("Run all tests (with and without LLMs)")]
[group("tests")]
test-all *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    test/ {{ params }}

[doc("Run all tests with coverage")]
[group("tests")]
test-all-cov *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    --cov=ag2 --cov-branch --cov-report=xml \
    test/ {{ params }}
  coverage report -m --include="ag2/*"


# Linter

[doc("Ruff check")]
[group("linter")]
ruff-check *params:
  ruff check {{ params }}

[doc("Ruff format")]
[group("linter")]
ruff-format *params:
  ruff format {{ params }}

[doc("Check typos (codespell + prek typos)")]
[group("linter")]
typos:
  prek run --all-files codespell
  prek run --all-files typos

[doc("Run ruff check + format")]
[group("linter")]
lint: ruff-check ruff-format typos
  prek run --all-files check-license-headers

[doc("Run zizmor on GitHub Actions workflows")]
[group("linter")]
zizmor *params:
  zizmor {{ params }} .

# Static analysis

[doc("Run mypy type check")]
[group("static analysis")]
mypy *params:
  mypy {{ params }}


# Prek

[doc("Install prek hooks")]
[group("prek")]
pre-commit-install:
  prek install

[doc("Run prek on modified files")]
[group("prek")]
pre-commit:
  prek run

[doc("Run prek on all files")]
[group("prek")]
pre-commit-all:
  prek run --all-files


# Docs

[doc("Build documentation")]
[group("docs")]
docs-build *params:
  cd website/mkdocs && python docs.py build {{ params }}

[doc("Serve documentation locally")]
[group("docs")]
docs-serve *params: docs-build
  cd website/mkdocs && python docs.py live {{ params }}
