#!/bin/bash
set -e
set -o xtrace

# We assume that the packages are already signed
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../common/init.sh
ARCH_REPO_DIR=/data/repos/archlinux/x86_64/

mkdir -p $ARCH_REPO_DIR

cd ${stage_dir["archlinux-rolling"]}
for f in *.tar.zst
do
	mv $f $ARCH_REPO_DIR
	mv $f.sig $ARCH_REPO_DIR
	repo-add --sign -k B2B75080 $ARCH_REPO_DIR/hipsycl.db.tar $ARCH_REPO_DIR/$f
done

