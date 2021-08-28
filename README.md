# Non-Euclidean Ray Trace Engine

## Demo
```
mkdir images
cd src
python demo.py
```
This script was used to create the frames of this video:

https://user-images.githubusercontent.com/7516208/131228594-f264acce-c461-4ec6-891b-0ef9b7262717.mp4

Here, the light rays are bend slightly towards a direction orthogonal to their position in space and their direction. The segments of the light rays are approximated with short straight lines.
Throughout the video the strength of this bending is varied and in the beginning no bending takes place.

## Run Tests
```
cd src
python run_tests.py
```

## Dependencies
- python 3.9+
- pip packages: numpy 1.19+, Pillow 8.3+
