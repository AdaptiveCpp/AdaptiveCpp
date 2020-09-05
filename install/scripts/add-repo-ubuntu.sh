#!/bin/bash

echo "deb http://repo.urz.uni-heidelberg.de/sycl/deb/ ./bionic main" > /etc/apt/sources.list.d/hipsycl.list
wget -q -O - http://repo.urz.uni-heidelberg.de/sycl/hipsycl.asc | apt-key add -
apt update

