#!/bin/bash

# This script sets up the Herwig environment for particle physics simulations.
# check if generators/herwig directory does not exist, if it is missing, download and install Herwig
if [ ! -d "generators/herwig" ]; then
    curl -L "https://herwig.hepforge.org/downloads?f=herwig-bootstrap" -o herwig-bootstrap
    chmod +x herwig-bootstrap
    ./herwig-bootstrap -j 4 $PWD/generators/herwig
fi

