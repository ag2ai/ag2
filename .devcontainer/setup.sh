# update pip
pip install --upgrade pip

# install dev packages
pip install -e ".[dev]"

# install pre-commit hook if not installed already
pre-commit install

# install Claude Code CLI
npm install --location=global @anthropic-ai/claude-code
