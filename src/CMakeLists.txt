subdirs(libhipSYCL)

include(ExternalProject)

# We compile these tools as external projects so they are
# able to use the default system compiler. They don't need
# to be compiled by the device compiler, so we can just use
# the regular one
ExternalProject_Add(hipsycl_rewrite_includes
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hipsycl_rewrite_includes
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL}
)

ExternalProject_Add(hipsycl_transform_source
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hipsycl_transform_source
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL}
)
