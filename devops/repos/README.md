# hipSYCL repo creation scripts

These scripts create the repositories for the supported distributions

## Operation
The new packages should be placed in the `stage/new_pkg_<distribution>` folder of the repository, after that the `create_pkgs.sh` script will move them to the repositories and sign them. If you want to have the repositories elsewhere, you can use the HIPSYCL_PKG_REPO_BASE_DIR environmental variable. The `create_repos.sh` script uses singularity containers to create the repositories for the distributions. The containers are only there to have the necessary build tools available regardless of the host operating system.  Therefore they only need to be built once, with the `create_singularity_containers.sh` script.

The workflow for manually adding packages:
*  `sh create_singularity_containers.sh`  
*  move packages to  `stage/new_pkg_<distribution>`   
*  execute `sh create_repos.sh`

with the `create_base_pkgs.sh` and the `create_hipsycl_pkgs.sh` the packages can be automatically built from the source, using the already available install scripts, and singularity containers. See the corresponding README file for more details.

If the `HIPSYCL_GPG_KEY` variable is set, then the archlinux packages will be singed. 
## Signing

The signing is carried out by the default private key of the user executing the `create_all.sh` script. 
 
