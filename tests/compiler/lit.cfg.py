import lit.formats
import os

config.name = 'hipSYCL Plugin'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.c', '.cpp', '.cc']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'test')

config.substitutions.append(('%syclcc', config.hipsycl_syclcc))

hipsycl_pipeline = "cbs"
if "HIPSYCL_PIPELINE" in os.environ:
  hipsycl_pipeline = os.environ["HIPSYCL_PIPELINE"]
config.environment["HIPSYCL_PIPELINE"] = hipsycl_pipeline

config.available_features.add(hipsycl_pipeline)