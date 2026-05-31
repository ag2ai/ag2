# update pip
pip install --upgrade pip

# install dev packages
pip install -e ".[dev]"

# install prek hook if not installed already
prek install

# install Claude Code CLI
npm install --location=global @anthropic-ai/claude-code
