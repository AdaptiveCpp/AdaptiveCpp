#!/bin/bash

add-apt-repository -y ppa:ubuntu-toolchain-r/test
echo "deb http://repo.urz.uni-heidelberg.de/sycl$1/deb/ ./bionic main" > /etc/apt/sources.list.d/hipsycl.list
wget -q -O - http://repo.urz.uni-heidelberg.de/sycl/hipsycl.asc | apt-key add -
apt update

