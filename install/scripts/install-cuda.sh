export HIPSYCL_INSTALL_PREFIX=${HIPSYCL_INSTALL_PREFIX:-/opt/hipSYCL}

CUDA_INSTALLER_FILENAME=cuda_10.0.130_410.48_linux

set -e
cd /tmp
if [ ! -f $CUDA_INSTALLER_FILENAME ]; then
  wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/$CUDA_INSTALLER_FILENAME
fi
sh $CUDA_INSTALLER_FILENAME --override --silent --toolkit --toolkitpath $HIPSYCL_INSTALL_PREFIX/cuda
