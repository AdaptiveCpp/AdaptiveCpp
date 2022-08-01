#!/bin/bash

# This small script is needed for the workflow to work
# since it is currently not possible to build containers
# inside containers, we break out from the container containing
# the workflow to a separate user on our server
#
# The GitHub Action sets the variables in its container
# we use this script to record the variables
# and then we copy the variables in a sourceable form
# to the user where the actual building will happen.

rm -rf envs.out
touch envs.out
for env in `env | grep HIPSYCL`; do
  echo "export $env" >> envs.out
done
