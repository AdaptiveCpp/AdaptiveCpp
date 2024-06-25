# hipSYCL packaging system

Currently the packaging is based around three groups of bash scripts bound together by the `update-repos.sh`, and the `common/init.sh scripts`. We aimed for having most of these scripts available for use separately from the packaging system, and to serve as inspiration.

The three logical groups are installation, package creation, repository creation, and testing. installation and package creation scripts are located in the `install/scripts` directory, repo creation and testing scripts are located in the `devops/repos directory`.

We provide a high level overview of the different functions here please refer to the actual scripts for more detail

## update_repo.sh

This script serves as a wrapper around the different other scripts that are responsible for building packaging and testing. It is usefulness lies in creating a access point for all the functions that are scattered among the different directories.

## record_env_vars.sh

Creates the `~/envs.out` file, based on the current environment.

## create_pkgs.sh

Executes the packaging script for a distro. and moves the finished packages to the staging folder. It has two modes, `hipsycl` and `base` the former only builds the hipSYCL packages later only builds the base packages

## create_repos.sh

Executes the repo creation for a distribution.

## test-packages.sh

Handles testing of the built and deployed packages for a certain backend configuration.

## test-installation.sh

Run tests on a singularity container containing hipSYCLs

## Examples

```
bash update_repo.sh centos-7 build_base build              # Build base container
bash update_repo.sh centos-7 build_base spack-install/rocm # Install rocm into base container
bash update_repo.sh centos-7 build_base spack-install/llvm # Install llvm
bash update_repo.sh centos-7 package base                  # create base packages for rocm and llvm&boost
bash update_repo.sh centos-7 package hipsycl               # create hipsycl packages
bash update_repo.sh centos-7 deploy                        # deploy packages
bash update_repo.sh centos-7 test 00                       # run build, add_repo install_dep run_test for the test
bash update_repo.sh centos-7 test 00 build                 # build testing container
bash update_repo.sh centos-7 test 00 add_repo               # Add hipSYCL repo to testing container
bash update_repo.sh centos-7 pub_cont                      # Publish containers 
```


 
