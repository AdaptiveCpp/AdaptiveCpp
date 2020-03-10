#!/bin/bash
#set -e

source ./common/init.sh

UBUNTU_REPO_DIR=${UBUNTU_REPO_DIR:-/data/repos/deb}
DIST=${DIST:-bionic}

PKG_PATH=$UBUNTU_REPO_DIR/dists/bionic/main/binary-amd64/
RELEASE_PATH=$UBUNTU_REPO_DIR/dists/$DIST/
POOL_PATH=$UBUNTU_REPO_DIR/pool/

mkdir -p $PKG_PATH
mkdir -p $POOL_PATH

cd $UBUNTU_PKG_DIR

for f in *
do
	echo $f
	mv $f $POOL_PATH

done
cd $UBUNTU_REPO_DIR
# we need the relative path because it will write it directly in Packages
apt-ftparchive  packages ./pool >  $PKG_PATH/Packages

cd $PKG_PATH
gzip -k -f $PKG_PATH/Packages
cd $RELEASE_PATH
echo `pwd`
apt-ftparchive release .  | tee $RELEASE_PATH/Release

echo `pwd`
rm Release.gpg
rm InRelease
gpg -abs -o Release.gpg Release
gpg --clearsign -o InRelease Release

