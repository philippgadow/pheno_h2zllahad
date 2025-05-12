#!/bin/bash

# docker pull gitlab-registry.cern.ch/atlas-sa/simple-analysis
docker run --rm -it -v $HOME:$HOME -w $PWD gitlab-registry.cern.ch/atlas-sa/simple-analysis