---
name: Bug report
about: Report a bug
title: ''
labels: bug
assignees: ''

---

**Bug summary**
A clear and concise description of what the problem is.

**To Reproduce**
Steps to reproduce the behavior. 

_If possible, provide a minimal reproducer_. The shorter the reproducing code snippet is, the easier will it be for us to debug and understand the issue.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Describe your setup**
* how you have installed AdaptiveCpp, and which version/branch/git commit
* Describe the dependencies that AdaptiveCpp sits on top of in your setup: 
   * clang version and how clang was installed
   * host compiler
   * CUDA version (if applicable)
   * ROCm version and how ROCm was installed (if applicable)
* How you have compiled your application and which arguments you have passed to `syclcc`. In particular, which backends and hardware you have compiled for.


**Optional additional diagnostic information**
The following information can potentially help us:
* For compilation/toolchain or setup-related issues: Setting the environment variable `HIPSYCL_DRYRUN=1` during compilation will cause `syclcc` to print the commands it would execute instead of actually executing. This can help verify the sanity of your setup.
* `syclcc --hipsycl-version` prints information about the AdaptiveCpp configuration that may be interesting for setup-related issues.
* For issues related to the runtime, setting the environment variable `ACPP_DEBUG_LEVEL=3` will cause it to print *a lot* of diagnostic information that might be helpful. Attach the output of your program with `ACPP_DEBUG_LEVEL=3` if you think it might be helpful for your issue.
* Recent AdaptiveCpp versions include a tool called `acpp-info`, which will print information about available backends and devices. This may be interesting for issues related to e.g. device visibility.

**Additional context**
Add any other context about the problem here.
