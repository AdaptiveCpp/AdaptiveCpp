add_executable(hipsycl-hcf-tool hipsycl-hcf-tool.cpp)
target_compile_definitions(hipsycl-hcf-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(hipsycl-hcf-tool PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)

install(TARGETS hipsycl-hcf-tool DESTINATION bin)
