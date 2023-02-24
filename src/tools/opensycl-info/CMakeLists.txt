add_executable(opensycl-info opensycl-info.cpp)

target_compile_definitions(opensycl-info PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(opensycl-info PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(opensycl-info PRIVATE hipSYCL-rt)
install(TARGETS opensycl-info DESTINATION bin)
