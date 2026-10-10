import gdb
import re

class AdaptiveCppVecPrinter:
    """Pretty Printer for AdaptiveCpp sycl::vec<DataT, NumElements>"""
    def __init__(self, val):
        self.val = val
        # Parse out Data Type and Element Count from the full template type string
        self.sycldatatype = str(val.type)
        # We understand a SYCL vector as in accordance with following regex  
        match = re.search(r'(?:\w+::)*sycl::vec<', self.sycldatatype)
        
        # Fetch template arguments  
        if match:
            template_pattern = re.compile(
                r"""(?:\w+::)*
                sycl::vec<
                    \s*
                    (?P<T>[^,<>]+)
                    \s*,\s*
                    (?P<N>\d+)
                    \s*,\s*
                    (?P<VectorStorage>
                        hipsycl::sycl::detail::vec_storage<
                            \s*[^,<>]+\s*,\s*\d+\s*
                        >
                    )
                    \s*>
                """,
                re.VERBOSE,
            )
            template_args = template_pattern.search(self.sycldatatype)
            if template_args:
                self.DataT = template_args.group("T")
                try:
                    self.NumElements = int(template_args.group("N"))
                except:
                    self.NumElements = 0
        else:
            self.DataT = None
            self.NumElements = 0

    def to_string(self):
        if self.NumElements == 0:
            return "sycl::vec (could not parse AdaptiveCpp dimensions)"
        
        # Resolve the internal storage member for AdaptiveCpp vs hipSYCL legacy
        m_data = None
        for field_candidate in ['_data', 'm_Data', 'data']:
            try:
                m_data = self.val[field_candidate]
                break
            except gdb.error:
                continue
        
        elements = []
        for i in range(self.NumElements):
            try:
                if m_data is not None:
                    elements.append(str(m_data['_storage'][i]))
            except gdb.error:
                elements.append("?")

        return f"sycl::vec<{self.DataT}, {self.NumElements}> = {{{', '.join(elements)}}}"
 

    def display_hint(self):
        return 'SYCL vector'

def lookup_adaptivecpp_printer(val):
    type_tag = str(val.type.unqualified())
    if 'sycl::vec<' in type_tag or 'sycl::_V1::vec<' in type_tag:
        return AdaptiveCppVecPrinter(val)
    return None

# Register with GDB session
gdb.pretty_printers.append(lookup_adaptivecpp_printer)
print("AdaptiveCpp sycl::vec pretty printer successfully initialized.")
