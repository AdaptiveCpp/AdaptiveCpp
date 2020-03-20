#!/usr/bin/env python3
import os
import os.path
import sys
import subprocess
import shutil
import json

def get_from_environment_or_default(variable_name, default_value):
  if variable_name in os.environ:
    return os.environ[variable_name]
  else:
    return default_value

singularity_executable = get_from_environment_or_default("SINGULARITY_PATH","/usr/bin/singularity")

def ensure_directory_exists(dirname):
  os.makedirs(dirname, exist_ok=True)

def raise_if_nonzero(command, environment={}):
  returncode = subprocess.call(command, env=environment, shell=True)
  if returncode != 0:
    raise RuntimeError("Command {} returned exit code {}".format(command,returncode))

def pull(user, branch):
  print("Using remote repository github.com/{}/hipSYCL:{}".format(user,branch))
  
  slug = user+"/hipSYCL"
  
  source_path = os.path.join("/tmp/",os.path.join(slug,branch))

  try:
    shutil.rmtree(source_path)
  except:
    pass

  ensure_directory_exists(source_path)
  subprocess.call("git clone --recurse-submodules -b  {} https://github.com/{} {}".format(
      branch, slug, source_path
      ),shell=True)
  
  return source_path


class stage1:

  def __init__(self, src_dir):
    self._src = src_dir
    self._failed_configs = []
  
  @property
  def all_configs(self):
    targets = ['ubuntu-18.04']
    
    return [{'build-os' : target} for target in targets]
  
  @property
  def failed_configs(self):
    return self._failed_configs
  
  def image_name(self, s1_config):
    job_name = "unnamed"
    
    if 'HIPSYCL_AUTO_TEST_JOBNAME' in os.environ:
      job_name = os.environ['HIPSYCL_AUTO_TEST_JOBNAME']
      
    for param in s1_params:
      job_name += '-' + s1_config[param]
      
    return "/tmp/hipsycl-auto-test-{}.sif".format(job_name)
  
  def run(self):
    configs = self.all_configs
    
    definition_files = {
      'ubuntu-18.04' : 'ubuntu-18.04.def'
    }
    
    for config in configs:
      image_path = image_name(config)
      # These variables will be exported to the singularity container build process.
      # Note: singularity makes variables of the form SINGULARITYENV_XYZ
      # available as XYZ inside the container.
      singularity_environment={
        'SRCPATH' : self._src,
        'SINGULARITYENV_SRCPATH' : self._src,
      }
    
      try:
        raise_if_nonzero("sudo {} build {} {}".format(singularity_executable, image_path, definition_files[config['build-os']]))
      except:
        print("S1 config failed:", config)
        self._failed_configs.append(config)
        
class stage2:
  
  def __init__(self, s1):
    self._stage1 = s1
    self._failed_configs = []

  def _make_config(self, test, platform, dev, with_runtime_testing):
    return {
      'test' : test,
      'platform' : platform,
      'device' : dev,
      'runtime-testing' : with_runtime_testing
    }
  
  @property
  def failed_configs(self):
    return self._failed_configs

  @property
  def all_configs(self):
    tests = ['./unit-tests.sh' : './sycl-pstl.sh']
    
    platforms = ['rocm', 'cuda', 'cpu']
    
    devs {
      'rocm' : ['gfx900', 'gfx906'],
      'cuda' : ['sm_52', 'sm_60', 'sm_70'],
      'cpu' : ['cpu']
    }
    runtime_testing = ['cpu'] + devs['rocm'] + devs['cuda']
    
    configs = []
    
    for test in tests:
      for platform in platforms:
        for dev in devs[platform]:
          configs.append(self._make_config(test, platform, dev, dev in runtime_testing))
    
    return configs
  
  def run(self):
    configs = self.all_configs
    
    s1_configs = self._stage1.all_configs
    for s1_config in self._s1_configs:
      image = self._stage1.image_name(s1_config)
      
      for config in configs:
        env = {
          'HIPSYCL_AUTO_TEST_PLATFORM' : config['platform'],
          'HIPSYCL_AUTO_TEST_GPU_ARCH' : config['device']
        }
        if config['runtime-testing']:
          env['HIPSYCL_AUTO_TEST_RUNTIME_TESTING'] = '1'
        else
          env['HIPSYCL_AUTO_TEST_RUNTIME_TESTING'] = '0'

        try:
          raise_if_nonzero("{} exec {} sh {}" .format(
                       singularity_executable,
                       image,
                       config['test']
                     ),
                     environment=env)
        except:
          print("S2 config failed:", config)
          self._failed_configs.append(config)

if __name__ == '__main__':
  
  s1_params, s2_params = get_test_config()
  
  source_path = ""
  if len(sys.argv) == 3:
    source_path = pull(sys.argv[1], sys.argv[2])
  else:
    print("Using local repository...")
    source_path = os.path.join(os.getcwd(),"..")
    job_build_path = os.path.join(build_path, "local")
  
  
  
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



