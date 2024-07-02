#!/bin/bash

## tag local

git tag -a $(python setup.py --version) -m ''

## push tag remote

git push origin --tags
