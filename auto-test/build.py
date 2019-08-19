#!/usr/bin/env python3
import os
import os.path
import sys
import subprocess

def get_from_environment_or_default(variable_name, default_value):
  if variable_name in os.environ:
    return os.environ[variable_name]
  else:
    return default_value
  
def ensure_directory_exists(dirname):
  os.makedirs(dirname, exist_ok=True)
  
def are_runtime_tests_allowed(platform_name):
  if "HIPSYCL_NO_RUNTIME_TESTING" in os.environ:
    disabled_platforms = os.environ["HIPSYCL_NO_RUNTIME_TESTING"].lower().split(',')
    if platform_name in disabled_platforms:
      return False
  return True
  
def raise_if_nonzero(command, environment={}):
  returncode = subprocess.call(command, env=environment, shell=True)
  if returncode != 0:
    raise RuntimeError("Command {} returned exit code {}".format(command,returncode))

if __name__ == '__main__':
  # The path where built images will be stored.
  build_path = os.path.join(os.getcwd(),"../../hipsycl-singularity-build")
  cuda_gpu_arch = get_from_environment_or_default("CUDA_GPU_ARCH", "sm_52")
  rocm_gpu_arch = get_from_environment_or_default("ROCM_GPU_ARCH", "gfx906")
  singularity_executable = get_from_environment_or_default("SINGULARITY_PATH","/usr/bin/singularity")
  # Path to the hipSYCL source code
  source_path = ""
  # Path where the final container image containing hipSYCL will be stored
  image_path = ""
  # Path where the base image will be stored. The base image contains
  # the required components to build hipSYCL (e.g. ROCm, CUDA, clang/llvm).
  # Storing the base image separately and building the hipSYCL image
  # on top of it allows for faster builds of subsequent invocations
  # of this script.
  base_image_path = os.path.join(build_path,"hipsycl-base.sif")
  
  # Path where this particular run will store its output
  job_build_path = ""
  
  if len(sys.argv) == 3:
    print("Using remote repository...")
    user = sys.argv[1]
    branch = sys.argv[2]
    
    slug = user+"/hipSYCL"
    
    source_path = os.path.join("/tmp/",os.path.join(slug,branch))
    job_build_path = os.path.join(build_path,"{}/{}".format(user,branch))
    
    ensure_directory_exists(source_path)
    subprocess.call("git clone --recurse-submodules -b  {} https://github.com/{} {}".format(
        branch, slug, source_path
      ),shell=True)
  else:
    print("Using local repository...")
    source_path = os.path.join(os.getcwd(),"..")
    job_build_path = os.path.join(build_path, "local")
  
  ensure_directory_exists(job_build_path)
  image_path = os.path.join(job_build_path,"hipsycl.sif")

  print("Building in {} from {}".format(job_build_path,source_path))
  

  # These variables will be exported to the singularity container build process.
  # Note: singularity makes variables of the form SINGULARITYENV_XYZ
  # available as XYZ inside the container.
  singularity_environment={
    'SRCPATH' : source_path,
    'SINGULARITYENV_SRCPATH' : source_path,
    # Target architectures - mostly relevant for the compilation of the tests
    'SINGULARITYENV_HIPSYCL_CUDA_GPU_ARCH' : cuda_gpu_arch,
    'SINGULARITYENV_HIPSYCL_ROCM_GPU_ARCH' : rocm_gpu_arch,
    # CMAKE_CXX_COMPILER used when compiling hipSYCL
    'SINGULARITYENV_HIPSYCL_CPU_CXX' : get_from_environment_or_default("HIPSYCL_CPU_CXX","clang++-10"),
    # CMAKE_C_COMPILER used when compiling hipSYCL
    'SINGULARITYENV_HIPSYCL_CPU_CC' : get_from_environment_or_default("HIPSYCL_CPU_CC","clang-10"),
    # cmake CLANG_EXECUTABLE_PATH used when compiling hipSYCL
    'SINGULARITYENV_HIPSYCL_CLANG' : get_from_environment_or_default("HIPSYCL_CLANG","/usr/bin/clang++-10"),
    # cmake LLVM_DIR used when compiling hipSYCL
    'SINGULARITYENV_HIPSYCL_LLVM_DIR' : get_from_environment_or_default("HIPSYCL_LLVM_DIR","/usr/lib/llvm-10/cmake"),
  }


  # If it doesn't exist yet, create the image containing
  # the hipSYCL dependencies and build environment.
  # Building this can take some time, so once it is built
  # it will be reused by subsequent hipSYCL builds.
  # Only delete it if something has changed in the base image
  # (e.g., new llvm/clang versions)
  if not os.path.isfile(base_image_path):
    print("""
===========================================

Building hipSYCL base image...
target location: {}

===========================================
""".format(base_image_path))
    raise_if_nonzero("sudo {} build {} hipsycl-env.def".format(singularity_executable, base_image_path))
    
  else:
    print("""
===========================================

Reusing base image since it already exists.
Delete:
{}
and rerun this script if you want to rebuild
the base image.

===========================================
""".format(base_image_path))

  # Like the base image, the hipSYCL image is not rebuilt
  # if it already exists.
  if not os.path.isfile(image_path):
    print("""
===========================================

Building hipSYCL container image...
source directory: {}
target: {}

===========================================
""".format(source_path, image_path))
    raise_if_nonzero("sudo -E {} build {} hipsycl.def".format(singularity_executable,image_path),
                     environment=singularity_environment)
  
  else:
    print("""
===========================================

Reusing hipSYCL image since it already exists.
Delete:
{}
and rerun this script if you want to rebuild
the image.

===========================================
""".format(image_path))

  #---------------------------------------------------
  
  print("""
===========================================
Starting tests...
===========================================
""")

  ensure_directory_exists(os.path.join(job_build_path, "external-tests"))

  print("""
-------------------------------------------
Running SYCL parallel STL compilation test...
-------------------------------------------
""")
  sycl_parallel_stl_path = os.path.join(job_build_path,"external-tests/SyclParallelSTL")
  if not os.path.exists(sycl_parallel_stl_path):
    raise_if_nonzero("git clone https://github.com/KhronosGroup/SyclParallelSTL {}".format(sycl_parallel_stl_path))

  platforms = {
    'cpu' : {
      'arch' : 'dummy-arch'
    },
    'cuda' : {
      'arch' : cuda_gpu_arch
    },
    'rocm' : {
      'arch' : rocm_gpu_arch
    }
  }


  for platform in platforms:
    target_arch = platforms[platform]['arch']

    test_environment = {
      'SINGULARITYENV_HIPSYCL_PLATFORM' : platform,
      'SINGULARITYENV_HIPSYCL_GPU_ARCH' : target_arch
    }
  
    def compile_parallel_stl_file(filename):
      print("Compiling SYCL parallel STL file {} for {}...".format(os.path.basename(filename), platform))
      raise_if_nonzero("singularity exec -B {} {} /usr/bin/syclcc-clang -o {} -I {} {}".format(
                       # Create a bind mount to let the container access the host file sysem
                       # where SYCL parallel STL is stored
                       sycl_parallel_stl_path,
                       image_path,
                       os.path.join(sycl_parallel_stl_path,os.path.basename(filename)+"."+platform),
                       os.path.join(sycl_parallel_stl_path,"include"),
                       os.path.join(sycl_parallel_stl_path,filename)
                     ),
                     environment=test_environment)
    
    compile_parallel_stl_file(os.path.join(sycl_parallel_stl_path, "examples/sycl_sample_00.cpp"))
    compile_parallel_stl_file(os.path.join(sycl_parallel_stl_path, "examples/sycl_sample_01.cpp"))

  print("""
-------------------------------------------
Running hipSYCL unit tests...
-------------------------------------------
""")

  for platform in platforms:
    if are_runtime_tests_allowed(platform):
      print("Running {} unit tests...".format(platform))
      raise_if_nonzero("singularity exec --nv {} /usr/share/hipSYCL/tests/{}/unit_tests".format(image_path,platform))
      print("=====> Passed")
    else:
      print("Skipping unit tests for {}".format(platform))



