add_executable(hipsycl-info hipsycl-info.cpp)

target_compile_definitions(hipsycl-info PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(hipsycl-info PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(hipsycl-info PRIVATE hipSYCL-rt)
install(TARGETS hipsycl-info DESTINATION bin)
