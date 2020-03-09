#!/bin/bash
set -e
CENTOS_PKG_DIR=${CENTOS_PKG_DIR:-./new_pkg_centos}
CENTOS_REPO_DIR=/data/repos/rpm/centos7

mkdir -p $CENTOS_REPO_DIR

cd $CENTOS_PKG_DIR
for f in *
do
	echo $f
	mv $f $CENTOS_REPO_DIR
	rpmsign --addsign $CENTOS_REPO_DIR/$f
done
createrepo $CENTOS_REPO_DIR
