#include "hipSYCL/glue/reflection.hpp"

void reflected_from_other_tu() {}

bool other_tu_resolves() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(
      &reflected_from_other_tu);
  return hipsycl::glue::reflection::resolve_function_name(
             &reflected_from_other_tu) != nullptr;
}
