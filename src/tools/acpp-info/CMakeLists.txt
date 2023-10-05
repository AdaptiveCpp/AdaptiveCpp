add_executable(acpp-info acpp-info.cpp)

target_compile_definitions(acpp-info PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(acpp-info PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(acpp-info PRIVATE hipSYCL-rt)
# Make sure that acpp-info uses compatible sanitizer flags for sanitized runtime builds
if(NOT ${HIPSYCL_RT_SANITIZE} STREQUAL "")
  target_link_libraries(acpp-info PRIVATE -fsanitize=${HIPSYCL_RT_SANITIZE})
  target_compile_options(acpp-info PRIVATE -fsanitize=${HIPSYCL_RT_SANITIZE})
endif()
install(TARGETS acpp-info DESTINATION bin)
