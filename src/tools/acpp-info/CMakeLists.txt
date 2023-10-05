add_executable(acpp-info acpp-info.cpp)

target_compile_definitions(acpp-info PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(acpp-info PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(acpp-info PRIVATE acpp-rt)
install(TARGETS acpp-info DESTINATION bin)
