#!/bin/bash
set -e
# We assume that the packages are already signed
ARCH_PKG_DIR=${ARCH_PKG_DIR:-`pwd`/new_pkg_arch/}
ARCH_REPO_DIR=${ARCH_REPO_DIR:-/data/repos/archlinux/x86_64/}

mkdir -p $ARCH_REPO_DIR

cd $ARCH_PKG_DIR
for f in *.tar.xz
do
	mv $f $ARCH_REPO_DIR
	mv $f.sig $ARCH_REPO_DIR
	repo-add --sign $ARCH_REPO_DIR/hipsycl.db.tar $ARCH_REPO_DIR/$f
done

