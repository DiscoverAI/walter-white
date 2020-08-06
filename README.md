# Walter White
[![CircleCI](https://circleci.com/gh/DiscoverAI/walter-white.svg?style=shield)](https://circleci.com/gh/DiscoverAI/walter-white)
[![GitHub license](https://img.shields.io/github/license/DiscoverAI/walter-white)](https://github.com/DiscoverAI/walter-white/blob/master/LICENSE)
> "Say my name."
>
> -- Walter White (Breaking Bad)

A generative neural network model for creating drugs.

## Install pipenv (if you don't have it):
```bash
brew install pyenv
```

Add the following to your `~/.bash_profile`
```bash
export PIPENV_IGNORE_VIRTUALENVS=1
$(eval pyenv init)
```

## Install Python 3.6 (version defined in Pipefile)
```bash
pyenv install 3.6.11
```
Note: Tensorflow does not work with Python > 3.6

## Install packages from Pipfile:
This also creates a virtual environment
```bash
pipenv install --dev
```
*Note:* it automatically picks the right Python version

## Run tests
```bash
pipenv run pytest
```

## Run linter
```bash
pipenv run lint
```

## Run tensorboard
```bash
pipenv run tensorboard
```

Will create a tensorboard instance at [http://localhost:6006/](http://localhost:6006/) 
