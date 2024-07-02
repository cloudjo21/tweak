#!/bin/bash

export PKG_NAME=tweak

pip install torch==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
python setup.py sdist upload -r internal

echo "${PKG_NAME} deployment completed!"
