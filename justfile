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
test-beta-llm mark=_beta_llm_default_mark *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    -m "{{ mark }}" \
    test/beta/ {{ params }}

[doc("Run beta tests with LLM and coverage (e.g. just test-beta-llm-cov openai)")]
[group("tests")]
test-beta-llm-cov mark=_beta_llm_default_mark *params:
  pytest --ff -vv --durations=10 --durations-min=1.0 \
    --cov=autogen/beta/config --cov-branch --cov-report=xml \
    -m "{{ mark }}" \
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
