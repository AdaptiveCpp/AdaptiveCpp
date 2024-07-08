

#include <ostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/sycl.hpp"



using namespace hipsycl::rt;

int main()
{
  hipsycl::rt::runtime_keep_alive_token rt;
  // OP: buffer Memory requirement 
  std::cout << "Dumping memory requirement buffers: " << std::endl << std::endl;
  hipsycl::rt::range<3> c(10, 0, 0);
  auto data_region_ptr = std::make_shared<buffer_data_region>(c, 10, 10);
  hipsycl::rt::id<3> offset(10, 0, 0);
  hipsycl::rt::range<3> range(100, 0, 0);
  std::size_t size = 30;
  buffer_memory_requirement test_buffer(data_region_ptr, offset, range, 
      hipsycl::sycl::access::mode::read_write,
      hipsycl::sycl::access::target::global_buffer);
  test_buffer.dump(std::cout);

  auto data_region_ptr_2 = std::make_shared<buffer_data_region>(c, 10, 10);
  hipsycl::rt::id<3> offset_2(5, 0, 0);
  hipsycl::rt::range<3> range_2(100, 0, 0);
  std::size_t size_2 = 14;
  buffer_memory_requirement test_buffer_2(data_region_ptr_2, offset_2, range_2, 
      hipsycl::sycl::access::mode::read,
      hipsycl::sycl::access::target::global_buffer);
  test_buffer_2.dump(std::cout);
  
  // Kernel Operation
  std::cout << std::endl << "Dumping Kernel Operation: " << std::endl << std::endl;        
  requirements_list reqs{rt.get()};
  hipsycl::common::auto_small_vector<std::unique_ptr<backend_kernel_launcher>>
      backend_kernel_list;
  std::string kernel_name = "test_kernel";
  hipsycl::glue::kernel_launcher_data launch_data;
  kernel_operation kernel_op(
      kernel_name.c_str(),
      hipsycl::rt::kernel_launcher{launch_data, std::move(backend_kernel_list)},
      reqs);
  kernel_op.dump(std::cout);


  // Memory Location 
  device_id test_device;
  hipsycl::rt::id<3> access_offset(1, 1, 1);
  hipsycl::rt::id<3> access_offset_2(2, 2, 2);
  memory_location mem_loc_1(test_device, access_offset, data_region_ptr);
  memory_location mem_loc_2(test_device, access_offset_2, data_region_ptr);
  //mem_loc_1.dump_short(std::cout);

  // Memcpy_operation
  std::cout << std::endl << "Dumping memory location " << std::endl << std::endl;
  memcpy_operation test_memcpy(mem_loc_1, mem_loc_2, range);
  test_memcpy.dump(std::cout);
 

  // DAG
  std::cout << "Dumping entire DAG: " << std::endl << std::endl;
  dag test_dag;
  execution_hints no_hint;
  hipsycl::rt::node_list_t no_requirements = {};
  //kernels
  auto node_ptr1 = std::make_shared<dag_node>(no_hint, 
                                    no_requirements, 
                                    std::unique_ptr<operation>(&kernel_op),
                                    rt.get());
  auto node_memcpy1 = std::make_shared<dag_node>(no_hint, 
                                    hipsycl::rt::node_list_t{ node_ptr1 }, 
                                    std::unique_ptr<operation>(&test_memcpy),
                                    rt.get());
  auto node_ptr2 = std::make_shared<dag_node>(no_hint, 
                                    hipsycl::rt::node_list_t{ node_ptr1 , node_memcpy1},
                                    std::unique_ptr<operation>(&kernel_op),
                                    rt.get());
  auto node_ptr3 = std::make_shared<dag_node>(no_hint, 
                                    hipsycl::rt::node_list_t{ node_ptr2, node_ptr1 },
                                    std::unique_ptr<operation>(&kernel_op),
                                    rt.get());
  //Memcpy
  

  test_dag.add_command_group(node_ptr1);
  test_dag.add_command_group(node_ptr2);
  test_dag.add_command_group(node_ptr3);
  test_dag.add_command_group(node_memcpy1);
  test_dag.dump(std::cout);

  std::cout << "***End of dumping tests***" << std::endl;
}
