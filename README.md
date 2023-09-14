# `enunlg`: an Extensible NLG library for Python

Goals:
* provide an environment for our team to build their neural network programming skills using Pytorch
* develop a modular codebase that students can use for projects and that researchers can use for baselines and as the basis for new NNLG systems
* serve as a starting point for collaborations on pipeline NNLG, WebNLG shared tasks, etc

We are starting by re-implementing existing seq2seq models (TGen, SC-LSTM, Neural Checklist) and adapting the implementations to each other.

Design Principles:
* 'engineering' experiments should be trivial (trying different random seeds, hidden layers, kinds of nonlinearities or reccurent cells, types of padding, etc)
* 'research' experiments should be easy (knowing which part of the code needs to be modified to incorporate a new architecture, kind of attention, large pre-trained model, language, domain, or something else)

## Project Structure

We are managing dependencies and virtual environments using [`poetry`](https://python-poetry.org/). Testing is done with [`pytest`](https://pytest.org/).

We will primarily use READMEs for documentation, though maybe the built-in wiki in GitHub will be helpful at some point.
We are using GitHub issues to track bugs and plan features and projects.

## Getting Started

0. Install `poetry` into your normal Python environment. `poetry` will manage dependencies and create a virtual environment for you.

       $ pip install poetry

1. Clone this repository.

       $ git clone git@github.com:ANONYMIZED/enlg.git

2. Edit `enlg/pyproject.toml`'s entry for `torch` to choose the version matching your environment (or find your own version of torch to use!)

3. `cd` to the root of the cloned repository and run `poetry install`, which will install everything in `pyproject.toml` unless you have a `poetry.lock` file already

       $ cd enlg
       $ poetry env use 3.9
       $ poetry install

> **Note**: if you run into problems with the version of Python available on your system, it is often easy to install another version of Python using your package manager. On Fedora we use the following command with `dnf`, but the equivalent for Ubuntu would use `apt-get` and I think you could also use `brew` on MacOS. You will need to tell poetry to use this version of Python then, with `poetry env use 3.9`.
>
>        $ sudo dnf install python39

4. Download the [E2E Challenge](https://www.macs.hw.ac.uk/InteractionLab/E2E/) dataset so you have something to test.

       $ ./scripts/fetch_e2e.bash
    
If you don't want to run a random bash script, the URL for fetching the data and the intended directory structure is documented in the script used above.

5. Run `poetry shell` in your terminal to activate the environment, after which you can run `python script/tgen.py` (for example) and it will use the virtual environment instance of Python with all the correct dependencies.

       $ poetry shell
       (enlg-SOMECHARS-py3.9) $ python scripts/tgen.py

> **Note**: `SOMECHARS` above represents a sequence of characters which is a hash generated by poetry provide a unique location for the virtual environment it creates.

This model takes about 1400 minutes (23 hours, 10 minutes) to train on an Intel i7-4790K rated at 4.4 GHz (runing Fedora 36 w/Linux kernel 5.19)

## Dev environments

- dev: running Python 3.9 on Fedora 33-36. Editing in PyCharm.
