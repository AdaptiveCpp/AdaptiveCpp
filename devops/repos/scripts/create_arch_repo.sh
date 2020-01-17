#!/bin/bash
set -e
# We assume that the packages are already signed
source ./common/init.sh
ARCH_REPO_DIR=/data/repos/archlinux/x86_64/

mkdir -p $ARCH_REPO_DIR

cd $ARCH_PKG_DIR
for f in *.tar.xz
do
	mv $f $ARCH_REPO_DIR
	mv $f.sig $ARCH_REPO_DIR
	repo-add --sign $ARCH_REPO_DIR/hipsycl.db.tar $ARCH_REPO_DIR/$f
done

