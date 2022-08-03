#! /bin/bash

# usage: syclcc-clang --help | ./cleanup_syclcchelp.sh
# output is in sylccout then and should be copied into using-hipsycl.md

sed "s/\[current value: .*\]/[current value: NOT SET]/g" > syclccout

install_root=`cat syclccout | grep -Po "Installation root: \K(.*)$"`
sed "s%$install_root%/install/path%g" -i syclccout
sed -e "s%\(Plugin LLVM version: \)[[:digit:]][[:digit:]]*\(, can accelerate CPU: \)[[:alpha:]][[:alpha:]]*%\1<version>\2<bool>%g" -i syclccout
sed -e "s%librt-backend-\w\w*.so%librt-backend-<name>.so%g" -i syclccout
