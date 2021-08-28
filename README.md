# Non-Euclidean Ray-Tracing Engine

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
Use the following script to execute the demo script.
The results can be found in the directory `images`.
```
#! /usr/bin/bash

# activate venv
source .venv/bin/activate

# ensure existence of output directory
mkdir -p images

# execute demo script
cd src
python demo.py
```
This script was used to create the frames of this video:

https://user-images.githubusercontent.com/7516208/131228594-f264acce-c461-4ec6-891b-0ef9b7262717.mp4

Here, the light rays are bend slightly towards a direction orthogonal to their position in space and their direction. The segments of the light rays are approximated with short straight lines.
Throughout the video the strength of this bending is varied and in the beginning no bending takes place.

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
