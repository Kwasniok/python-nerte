# Non-Euclidean Ray-Tracing Engine
[![unittest](https://github.com/Kwasniok/python-nerte/actions/workflows/unittest.yml/badge.svg)](https://github.com/Kwasniok/python-nerte/actions/workflows/unittest.yml)
[![lint-pylint](https://github.com/Kwasniok/python-nerte/actions/workflows/pylint.yml/badge.svg)](https://github.com/Kwasniok/python-nerte/actions/workflows/pylint.yml)
[![lint-mypy](https://github.com/Kwasniok/python-nerte/actions/workflows/mypy.yml/badge.svg)](https://github.com/Kwasniok/python-nerte/actions/workflows/mypy.yml)

## Installation & Dependencies
NERTE requires [Python 3.9+](https://www.python.org/downloads/) with [pip](https://pip.pypa.io/en/stable/user_guide/) and the usage of a virtual environment ([venv](https://docs.python.org/3/tutorial/venv.html)) is recommended.

Execute these commands to install the dependencies:
```
#! /usr/bin/bash

# create virtual environment for nerte
python -m venv .venv

# activate venv
source .venv/bin/activate

# optional: verification
which python
which pip
# both should return paths located within the .venv/bin directory

# install dependencies
pip install -r requirements.txt
```
For a detailed list of dependencies see [this](requirements.txt) file.

## Run
- Before running any python script, enable the virtual environment first via the command `source .venv/bin/activate`.
The virtual environment is then active for the current session. Use `deactivate` to leave the virtual environment.
- All python scripts are designed be executed from within the `src` folder.

Example to run the script `main.py`:
```
#! /usr/bin/bash

# activate venv
source .venv/bin/activate

# execute script
cd src
python main.py
```

## Demo
Use the following script to execute any of the demo scripts.
The results can be found in the directory `images`.
```
#! /usr/bin/bash

# activate venv
source .venv/bin/activate

# ensure existence of output directory
mkdir -p images

# execute demo scripts
cd src
python demo_1.py
python demo_2.py
python demo_3.py
python demo_4.py
```

A similar script to `demo_4.py` was used to create the frames of this video:

https://user-images.githubusercontent.com/7516208/142043357-6c43a7a9-0b8a-492a-92d0-40de5174030c.mp4

In this demo video, the geodesics are bend into non-straight lines.
Throughout the video the strength of this bending is varied and in the middle of the video the space is flat.
The algorithm operates numerically and approximates a light ray with short striaght segments.
All segments are obtained via the Runge-Kutta algorithm based on the previous segment.

## Run Tests
The unittests are performed by this script:
```
#! /usr/bin/bash

# activate venv
source .venv/bin/activate

# execute unittests
cd src
python run_tests.py
```
