add_executable(acpp-pcuda-pp acpp-pcuda-pp.cpp)

target_compile_definitions(acpp-pcuda-pp PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(acpp-pcuda-pp PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)

install(TARGETS acpp-pcuda-pp DESTINATION bin)
