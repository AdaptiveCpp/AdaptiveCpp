#!/bin/bash
set -e
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../common/init.sh

UBUNTU_REPO_DIR=${UBUNTU_REPO_DIR:-/data/repos/deb}
DIST=${1:-bionic}

PKG_PATH=$UBUNTU_REPO_DIR/dists/$DIST/main/binary-amd64/
RELEASE_PATH=$UBUNTU_REPO_DIR/dists/$DIST/
POOL_PATH=$UBUNTU_REPO_DIR/pool/

mkdir -p $PKG_PATH
mkdir -p $POOL_PATH

cd ${stage_dir["ubuntu-18.04"]}

for f in *
do
	echo $f
	set +e
	mv $f $POOL_PATH
	set -e
done
cd $UBUNTU_REPO_DIR 
# we need the relative path because it will write it directly in Packages
apt-ftparchive  packages ./pool >  $PKG_PATH/Packages

cd $PKG_PATH
gzip -k -f $PKG_PATH/Packages || true
cd $RELEASE_PATH
echo `pwd`
apt-ftparchive release .  | tee $RELEASE_PATH/Release

echo `pwd`
rm -f Release.gpg
rm -f InRelease
gpg --batch --no-tty --default-key B2B75080 -abs -o Release.gpg Release
gpg --batch --no-tty --default-key B2B75080 --clearsign -o InRelease Release

