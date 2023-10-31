*Thank you for your interest in contributing to AdaptiveCpp! Your PR will be highly appreciated* :-) 

When in doubt about how to implement something or how to integrate with the overall project strategy, please just open an issue for discussion.

# Important branches in AdaptiveCpp

Please take note of the branch structure of the project. The following are important branches:

* `stable` - contains latest AdaptiveCpp with additional testing. `stable` should always contain a version of AdaptiveCpp that we are confident is stable.
* `develop` - contains latest development version.
* `sycl/VERSION` - contains AdaptiveCpp code that targets a specific SYCL version.
   - `sycl/1.2.1` - contains latest AdaptiveCpp targeting SYCL 1.2.1. This branch is now mainly in maintenance mode. If you want to specifically improve AdaptiveCpp SYCL 1.2.1 support, please use this branch.
   - `sycl/2020` - contains latest AdaptiveCpp work targeting SYCL 2020, and any work that is not specific to earlier SYCL versions.

We periodically perform the following merges:
* `develop` -> `sycl/<latest-version>` -> `stable`


Please follow the following guidelines:
* **File your PR against the `develop` branch, unless you are specifically targeting an earlier SYCL version.**
* **If you are targeting an earlier SYCL version, target the appropriate `sycl/<version>` branch**
   
