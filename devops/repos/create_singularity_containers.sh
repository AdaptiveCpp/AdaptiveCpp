#!/bin/bash
SINGULARITY_BASE_DIR=${SINGULARITY_BASE_DIR:-./containers/}
SINGULARITY_DEF_DIR=${SINGULARTIY_DEF_DIR:-./definitions-packaging-container/}

singularity build --fakreoot  $SINGULARITY_BASE_DIR/centos-7.sif  \
	 $SINGULARITY_DEF_DIR/base-centos-7.def

singularity build --fakreoot $SINGULARITY_BASE_DIR/ubuntu-18.04.sif  \
  	 $SINGULARITY_DEF_DIR/base-ubuntu-18.04.def

singularity build --fakreoot $SINGULARITY_BASE_DIR/arch.sif    \
	 $SINGULARITY_DEF_DIR/base-archlinux-rolling.def
