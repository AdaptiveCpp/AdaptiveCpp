#!/bin/bash
# checks whether __cxa_throw is present in device bitcode of specified HCF file (first arg)
# additionally checks if __acpp_sscp_assert_fail is present in device bitcode
# For the LLVM Integrated Tester (lit) respective patterns CXATHROWHIT and ACPPASSERTFAILHIT are printed to stdout

HCFFILE="$1"

# acpp-hcf tool path is set via CMakeLists in tests/compiler/
ACPP_HCF="$ACPP_COMPILER-hcf-tool"

# llvm version as env var from Dockerfile in workflow
LLVMNM="/usr/lib/llvm-$LLVM_VERSION/bin/llvm-nm"


SYMBOLS=$($LLVMNM -B --demangle <($ACPP_HCF $HCFFILE -x root images llvm-ir.global))

if [[ "$SYMBOLS" == *"cxa_throw"* ]]; then
    echo "CXATHROWHIT: HCF binary appendix (device bitcode) contains symbol __cxa_throw"
else 
    echo "No symbol __cxa_throw HCF binary appendix (device bitcode)."
fi

if [[ "$SYMBOLS" == *"acpp_sscp_assert_fail"* ]]; then
    echo "ACPPASSERTFAILHIT: HCF binary appendix (device bitcode) contains symbol __acpp_sscp_assert_fail"
else 
    echo "No symbol __acpp_sscp_assert_fail in HCF binary appendix (device bitcode)."
fi

