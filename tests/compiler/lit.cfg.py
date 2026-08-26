import lit.formats
import os
import platform

config.name = 'AdaptiveCpp Plugin'
config.test_format = lit.formats.ShTest()

config.suffixes = ['.c', '.cpp', '.cc']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root)

config.substitutions.append(('%acpp', config.acpp_compiler))

system = platform.system().lower()
if system == 'darwin':
  config.available_features.add('system-darwin')
elif system == 'linux':
  config.available_features.add('system-linux')
elif system == 'windows':
  config.available_features.add('system-windows')

if "ACPP_DEBUG_LEVEL" in os.environ:
  config.environment["ACPP_DEBUG_LEVEL"] = os.environ["ACPP_DEBUG_LEVEL"]
if "ACPP_HCF_DUMP_DIRECTORY" in os.environ:
  config.environment["ACPP_HCF_DUMP_DIRECTORY"] = os.environ["ACPP_HCF_DUMP_DIRECTORY"]

if "ACPP_VISIBILITY_MASK" in os.environ:
  mask = os.environ["ACPP_VISIBILITY_MASK"]
  config.environment["ACPP_VISIBILITY_MASK"] = mask

  # If a user has set the visibility mask env var then we set each backend as a
  # LIT feature so that tests can be marked as UNSUPPORTED based on backend.
  backend_masks = mask.split(';')
  backends = [ b.split(':')[0] for b in backend_masks]
  for b in backends:
    config.available_features.add(b)
