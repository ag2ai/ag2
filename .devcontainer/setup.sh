# update pip
pip install --upgrade pip

# install pre-release of quarto for docs
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.7/quarto-1.7.7-linux-amd64.deb -O quarto.deb && \
    sudo dpkg -i quarto.deb && \
    rm quarto.deb

# install dev packages
pip install -e ".[dev]"

# install pre-commit hook if not installed already
pre-commit install
