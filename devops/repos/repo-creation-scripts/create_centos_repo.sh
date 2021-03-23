#!/bin/bash
set -e
set -o xtrace
if [ "$#" -ne "1" ]; then
	echo "Please specifiy the distro (centos7 or cnetos8) as first argument"
fi
distro=$1

declare -A repo_dir=( ["centos-7"]="centos7" \
                      ["centos-8"]="centos8" \
					  )
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../common/init.sh
CENTOS_REPO_DIR=/data/repos/rpm/${repo_dir[$distro]}
mkdir -p $CENTOS_REPO_DIR
cd ${stage_dir[$distro]}
echo $HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
for f in *
do
	echo $f
	mv $f $CENTOS_REPO_DIR
	echo "" | setsid rpmsign --addsign $CENTOS_REPO_DIR/$f
done
createrepo $CENTOS_REPO_DIR
cp $DIR/hipsycl-$distro.repo $CENTOS_REPO_DIR/hipsycl.repo
echo $DIR
echo $CENTOS_REPO_DIR
sed "s|sycl{}|sycl$HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX|" $DIR/hipsycl-$distro.repo > $CENTOS_REPO_DIR/hipsycl.repo
