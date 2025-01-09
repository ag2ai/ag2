# update pip
pip install --upgrade pip

# install dev packages
pip install -e ".[test,teachable,lmm,retrievechat,mathchat,blendsearch]"

# install pre-commit hook if not installed already
pre-commit install
