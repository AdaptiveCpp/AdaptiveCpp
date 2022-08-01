# -*- coding: utf-8 -*-
import lit.formats
import os

config.name = 'hipSYCL Plugin'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.c', '.cpp', '.cc']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'test')

config.substitutions.append(('%syclcc', config.hipsycl_syclcc))

if "HIPSYCL_DEBUG_LEVEL" in os.environ:
  config.environment["HIPSYCL_DEBUG_LEVEL"] = os.environ["HIPSYCL_DEBUG_LEVEL"]
