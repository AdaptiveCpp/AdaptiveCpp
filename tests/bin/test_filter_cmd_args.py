'''
Regression test for filter_cmd_args() in bin/acpp, used by
--acpp-dryrun-only-std-flags to print a compile command stripped down to
"standard" compiler flags (includes, defines, warnings, -std=).

Run with: python3 tests/bin/test_filter_cmd_args.py
'''

import importlib.machinery
import importlib.util
import os
import sys
import unittest

ACPP_DRIVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "bin", "acpp")


def load_acpp_driver():
  loader = importlib.machinery.SourceFileLoader("acpp_driver", ACPP_DRIVER_PATH)
  spec = importlib.util.spec_from_loader("acpp_driver", loader)
  module = importlib.util.module_from_spec(spec)
  loader.exec_module(module)
  return module


acpp_driver = load_acpp_driver()


class TestFilterCmdArgs(unittest.TestCase):
  def test_separate_token_include_path_is_kept(self):
    # -I and its path are commonly forwarded as two separate argv entries,
    # e.g. when a Makefile expands "-I $(dir)" rather than "-I$(dir)".
    command = ["clang++", "-I", "/usr/local/include", "-DFOO=1", "-c",
               "foo.cpp", "-o", "foo.o"]
    result = acpp_driver.filter_cmd_args(command)
    self.assertIn("/usr/local/include", result)
    self.assertIn("-I", result)

  def test_glued_include_path_is_kept(self):
    command = ["clang++", "-I/usr/local/include", "foo.cpp"]
    result = acpp_driver.filter_cmd_args(command)
    self.assertIn("-I/usr/local/include", result)

  def test_separate_token_define_is_kept(self):
    # -D also accepts its value as a separate argument, e.g. "-D FOO=1".
    command = ["clang++", "-D", "FOO=1", "foo.cpp"]
    result = acpp_driver.filter_cmd_args(command)
    self.assertIn("FOO=1", result)
    self.assertIn("-D", result)

  def test_glued_define_is_kept(self):
    command = ["clang++", "-DFOO=1", "foo.cpp"]
    result = acpp_driver.filter_cmd_args(command)
    self.assertIn("-DFOO=1", result)

  def test_non_whitelisted_flag_is_removed(self):
    command = ["clang++", "-fplugin=/path/to/plugin.so", "foo.cpp"]
    result = acpp_driver.filter_cmd_args(command)
    self.assertNotIn("-fplugin=/path/to/plugin.so", result)


if __name__ == "__main__":
  unittest.main()
