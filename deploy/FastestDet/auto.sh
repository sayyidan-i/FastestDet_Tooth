#!/bin/bash

fswebcam test.jpg
python3.9 runtime.py --img test.jpg
