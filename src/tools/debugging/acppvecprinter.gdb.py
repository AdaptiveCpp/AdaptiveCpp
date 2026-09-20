import gdb
import re

class AdaptiveCppVecPrinter:
    """Pretty Printer for AdaptiveCpp sycl::vec<DataT, NumElements>"""
    def __init__(self, val):
        self.val = val
        # Parse out Data Type and Element Count from the full template type string
        type_str = str(val.type.unqualified())
        # Handles standard sycl namespaces and AdaptiveCpp variations
        match = re.search(r'sycl::(?:_V1::)?vec<\s*([^,]+)\s*,\s*(\d+)\s*>', type_str)
        
        if match:
            self.data_type = match.group(1)
            self.num_elements = int(match.group(2))
        else:
            self.data_type = None
            self.num_elements = 0

    def to_string(self):
        if self.num_elements == 0:
            return "sycl::vec (could not parse AdaptiveCpp dimensions)"
        
        # Resolve the internal storage member for AdaptiveCpp vs hipSYCL legacy
        m_data = None
        for field_candidate in ['_data', 'm_Data', 'data']:
            try:
                m_data = self.val[field_candidate]
                break
            except gdb.error:
                continue

        # If AdaptiveCpp falls back entirely on raw compiler built-in vector extensions,
        # treating the variable as an array often lets GDB pull indices natively.
        elements = []
        for i in range(self.num_elements):
            try:
                if m_data is not None:
                    elements.append(str(m_data[i]))
                else:
                    # Fallback to direct indexing on the variable itself if using compiler vector primitives
                    elements.append(str(self.val[i]))
            except gdb.error:
                elements.append("?")

        return f"sycl::vec<{self.data_type}, {self.num_elements}> = {{{', '.join(elements)}}}"

    def display_hint(self):
        return 'array'

def lookup_adaptivecpp_printer(val):
    type_tag = str(val.type.unqualified())
    if 'sycl::vec<' in type_tag or 'sycl::_V1::vec<' in type_tag:
        return AdaptiveCppVecPrinter(val)
    return None

# Register with GDB session
gdb.pretty_printers.append(lookup_adaptivecpp_printer)
print("AdaptiveCpp sycl::vec pretty printer successfully initialized.")
