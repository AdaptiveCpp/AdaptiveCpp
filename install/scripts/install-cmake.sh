export INSTALL_PREFIX=${INSTALL_PREFIX:-/opt/cmake}

CMAKE_INSTALLER_FILENAME=cmake-3.16.8-Linux-x86_64.sh

set -e
cd /tmp
if [ ! -f $CMAKE_INSTALLER_FILENAME ]; then
  wget https://github.com/Kitware/CMake/releases/download/v3.16.8/$CMAKE_INSTALLER_FILENAME
fi
sh $CMAKE_INSTALLER_FILENAME --skip-license --prefix=$INSTALL_PREFIX/cmake
