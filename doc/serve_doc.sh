#!/bin/sh

SCRIPTDIR=$(dirname "$0")

PYENVDIR=$SCRIPTDIR/../AdaptiveCppEnv

cd $SCRIPTDIR/..

if [ ! -d $PYENVDIR ]; then
  echo "$PYENVDIR does not exist, creating the venv"
  python3 -m venv AdaptiveCppEnv
fi

source $SCRIPTDIR/../AdaptiveCppEnv/bin/activate

pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-git-authors-plugin

mkdocs serve