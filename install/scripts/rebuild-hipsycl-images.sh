#!/bin/bash
sudo singularity build -F hipsycl-ubuntu-18.04.sif hipsycl-ubuntu-18.04.def
sudo singularity build -F hipsycl-archlinux-rolling.sif hipsycl-archlinux-rolling.def
sudo singularity build -F hipsycl-centos-7.sif hipsycl-centos-7.def

