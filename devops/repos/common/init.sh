#!/bin/bash

HIPSYCL_PKG_REPO_STAGE_DIR=${HIPSYCL_PKG_REPO_STAGE_DIR:-./stage}
export HIPSYCL_GPG_KEY=E967BA09716F870320089583E68CC4B9B2B75080

#Testing
declare -A install_cmd=( ["archlinux-rolling"]="pacman -Sy --noconfirm hipSYCL" \
                         ["centos-7"]="yum -y install hipSYCL" \
                         ["centos-8"]="yum -y install hipSYCL" \
                         ["ubuntu-18.04"]="apt -y install hipsycl" \
                         ["ubuntu-20.04"]="apt -y install hipsycl"
                       )

declare -A cleanup_cmd=( ["archlinux-rolling"]="pacman -Rsn --noconfirm hipSYCL" \
                         ["centos-7"]="yum -y remove hipSYCL" \
                         ["centos-8"]="yum -y remove hipSYCL" \
                         ["ubuntu-18.04"]="apt -y remove hipsycl" \
                         ["ubuntu-20.04"]="apt -y remove hipsycl"
                       )

declare -A cleanup_dep=( ["archlinux-rolling"]='pacman -Rsn --noconfirm $(pacman -Qdtq)' \
                         ["centos-7"]="package-cleanup -y --leaves" \
                         ["centos-8"]="package-cleanup -y --leaves" \
                         ["ubuntu-18.04"]="apt -y autoremove" \
                         ["ubuntu-20.04"]="apt -y autoremove"
                       )


declare -A image_base=( ["archlinux-rolling"]="docker://archlinux:base" \
                         ["centos-7"]="docker://centos:centos7" \
                         ["centos-8"]="docker://centos:centos8" \
                         ["ubuntu-18.04"]="docker://ubuntu:18.04" \
                         ["ubuntu-20.04"]="docker://ubuntu:20.04"
                      )

declare  -A pkg_suffix=( ["ONON"]="-full" ["OFFOFF"]="-omp" ["OFFON"]="-cuda" \
                         ["ONOFF"]="-rocm")

#Packging


declare -A find_built_pkg=( ["archlinux-rolling"]='4.pkg.tar' \
                            ["centos-7"]='4.rpm'  \
                            ["centos-8"]='4.rpm'  \
                            ["ubuntu-18.04"]='\.deb' \
                      )
declare -A packaging_script=( ["archlinux-rolling"]="make-archlinux-pkg.sh"    \
                              ["centos-7"]="make-centos-7-pkg.sh"  \
                              ["centos-8"]="make-centos-8-pkg.sh"  \
                              ["ubuntu-18.04"]="make-ubuntu-pkg.sh"  \
                      )
declare -A packaging_image=( ["archlinux-rolling"]="archlinux-rolling"    \
                              ["centos-7"]="centos-7"  \
                              ["centos-8"]="centos-8"  \
                              ["ubuntu-18.04"]="ubuntu-18.04"  \
                              )

declare -A stage_dir=( ["archlinux-rolling"]="$HIPSYCL_PKG_REPO_STAGE_DIR/new_pkg_arch"    \
                       ["centos-7"]="$HIPSYCL_PKG_REPO_STAGE_DIR/new_pkg_centos-7"  \
                       ["centos-8"]="$HIPSYCL_PKG_REPO_STAGE_DIR/new_pkg_centos-8"  \
                       ["ubuntu-18.04"]="$HIPSYCL_PKG_REPO_STAGE_DIR/new_pkg_ubuntu"  \
                      )

#Repo creation
declare -A repo_tools_cont=( ["archlinux-rolling"]="arch.sif" \
                             ["centos-7"]="centos-7.sif" \
                             ["centos-8"]="centos-7.sif" \
                             ["ubuntu-18.04"]="ubuntu-18.04.sif" \
                             ["ubuntu-20.04"]="ubuntu-18.04.sif"
                       )

declare -A repo_script=( ["archlinux-rolling"]="create_arch_repo.sh" \
                             ["centos-7"]="create_centos_repo.sh centos-7" \
                             ["centos-8"]="create_centos_repo.sh centos-8" \
                             ["ubuntu-18.04"]="create_ubuntu_repo.sh bionic" \
                             ["ubuntu-20.04"]="create_ubuntu_repo.sh focal"
                       )



#distros=( "centos-7" "archlinux-rolling" "ubuntu-18.04" "ubuntu-20.04")
#build_distros=( "centos-7" "archlinux-rolling" "ubuntu-18.04" )
