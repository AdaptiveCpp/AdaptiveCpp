add_executable(opensycl-hcf-tool opensycl-hcf-tool.cpp)
target_compile_definitions(opensycl-hcf-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(opensycl-hcf-tool PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)

install(TARGETS opensycl-hcf-tool DESTINATION bin)
