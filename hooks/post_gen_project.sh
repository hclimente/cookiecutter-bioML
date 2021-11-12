#!/bin/bash

rm */.gitkeep

git init .
git add .
git add -f src/templates/data
git commit -m "Initial setup"

pre-commit install
