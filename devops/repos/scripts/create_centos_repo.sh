#!/bin/bash
set -e
source ./common/init.sh

CENTOS_REPO_DIR=/data/repos/rpm/centos7

mkdir -p $CENTOS_REPO_DIR

cd $CENTOS_PKG_DIR
for f in *
do
	echo $f
	mv $f $CENTOS_REPO_DIR
	echo "" | setsid rpmsign --addsign $CENTOS_REPO_DIR/$f
done
createrepo $CENTOS_REPO_DIR
