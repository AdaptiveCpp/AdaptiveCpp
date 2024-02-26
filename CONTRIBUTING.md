*Thank you for your interest in contributing to AdaptiveCpp! Your PR will be highly appreciated* :-) 

When in doubt about how to implement something or how to integrate with the overall project strategy, please just open an issue for discussion.

# Important branches in AdaptiveCpp

Please take note of the branch structure of the project. The following are important branches:

* `stable` - contains latest AdaptiveCpp release.
* `develop` - contains latest development version.
* `sycl/VERSION` - contains AdaptiveCpp code that targets a specific SYCL version.
   - `sycl/1.2.1` - contains latest AdaptiveCpp targeting SYCL 1.2.1. This branch is now mainly in maintenance mode and not updated anymore.
   - `sycl/2020` - contains latest AdaptiveCpp work targeting SYCL 2020, and any work that is not specific to earlier SYCL versions.

We periodically perform the following merges:
* `develop` -> `sycl/<latest-version>` -> `stable`


Please files PRs against the `develop` branch.

# Release schedule

AdaptiveCpp follows a regular release schedule with 4-month cadence:

* `year.02` (released at the end of February)
* `year.06` (released at the end of June)
* `year.10` (released at the end of October)

In the last week prior to release, no new features should be added and focus should be on bug fixes.
In the last two weeks prior to release, new features should only be added for components of the stack that are still under heavy development (e.g. the SSCP compiler or stdpar support).
   
