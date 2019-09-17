#!/bin/bash
echo "Building Ubuntu 18.04 image..."
sudo singularity build -F base-ubuntu-18.04.sif base-ubuntu-18.04.def
echo "Building Arch Linux image..."
sudo singularity build -F base-archlinux-rolling.sif base-archlinux-rolling.def
echo "Building CentOS 7 image..."
sudo singularity build -F base-centos-7.sif base-centos-7.def
