#!/bin/bash
yum update -y
yum install epel-release -y
yum install -y rpm-build sed wget curl patch
yum install centos-release-scl -y
yum-config-manager --add-repo http://repo.urz.uni-heidelberg.de/sycl$1/rpm/centos7/hipsycl.repo
