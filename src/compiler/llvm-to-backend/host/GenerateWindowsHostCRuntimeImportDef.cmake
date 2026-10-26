if(NOT EXISTS "${UCRT_DLL}")
  message(FATAL_ERROR "Could not find UCRT DLL: ${UCRT_DLL}")
endif()

execute_process(
  COMMAND "${LLVM_READOBJ}" --coff-exports "${UCRT_DLL}"
  RESULT_VARIABLE READOBJ_RESULT
  OUTPUT_VARIABLE READOBJ_OUTPUT
  ERROR_VARIABLE READOBJ_ERROR
)

if(NOT READOBJ_RESULT EQUAL 0)
  message(FATAL_ERROR "llvm-readobj failed with exit code ${READOBJ_RESULT}:\n${READOBJ_ERROR}")
endif()

string(REPLACE "\r\n" "\n" READOBJ_OUTPUT "${READOBJ_OUTPUT}")
string(REPLACE "\n" ";" READOBJ_LINES "${READOBJ_OUTPUT}")

set(UCRT_EXPORTS)

foreach(LINE IN LISTS READOBJ_LINES)
  if(LINE MATCHES "^[ \t]*Name:[ \t]*(.+)$")
    list(APPEND UCRT_EXPORTS "${CMAKE_MATCH_1}")
  endif()
endforeach()

list(REMOVE_DUPLICATES UCRT_EXPORTS)
list(SORT UCRT_EXPORTS)

set(DEF_CONTENT "LIBRARY ucrtbase.dll\n\nEXPORTS\n")

foreach(SYMBOL IN LISTS UCRT_EXPORTS)
  string(APPEND DEF_CONTENT "    ${SYMBOL}\n")
endforeach()

file(WRITE "${OUTPUT_DEF}" "${DEF_CONTENT}")

list(LENGTH UCRT_EXPORTS EXPORT_COUNT)
message(STATUS "Generated UCRT import definition with ${EXPORT_COUNT} exports")
