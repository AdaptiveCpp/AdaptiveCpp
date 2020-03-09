# hipSYCL repo creation scripts

These scripts create the repositories for the supported distributions

## Operation

The new packages should be placed in the `new_pkg_<distribution>` folder of the repository, after that the create_all.sh script will move them to the repositories and sign them. if you want to have the repositories elsewhere, you can use the `REPO_BASE_DIR` environmental variable

## Signing

The signing is carried out by the default private key of the user executing the `create_all.sh` script. 
 
