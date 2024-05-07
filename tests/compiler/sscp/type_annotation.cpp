// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s


#include <iostream>

#include <limits>
#include <sycl/sycl.hpp>
#include "common.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

template<class T>
struct unannotated_wrapper {
  T value;
};

template <class T>
class __hipsycl_sscp_emit_param_type_annotation_custom_annotation1 {
  T value;
};

template <class T>
class __hipsycl_sscp_emit_param_type_annotation_specialized {
  T value;
};

#define make_test_kernel(name, argT)                                           \
  class name {                                                                 \
  public:                                                                      \
    name(int *data, argT arg) : _data{data} {}                                 \
    void operator()() const { *_data += 1; };                                   \
                                                                               \
  private:                                                                     \
    int *_data;                                                                \
    argT _arg;                                                                 \
  };

make_test_kernel(__test_kernel1_, unannotated_wrapper<int>);
make_test_kernel(__test_kernel2_,
                 __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<int>);
make_test_kernel(__test_kernel3_,
                 __hipsycl_sscp_emit_param_type_annotation_specialized<int>);
make_test_kernel(
    __test_kernel4_,
    __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
        __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<int>>);
make_test_kernel(
    __test_kernel5_,
    __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
        __hipsycl_sscp_emit_param_type_annotation_specialized<int>>);

using aggregate_type = std::pair<int, int>;
make_test_kernel(__test_kernel6_,
                 __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
                     aggregate_type>);

int main(int argc, char** argv)
{
  bool always_false = argc == std::numeric_limits<int>::max();
  auto hcf_id = __hipsycl_local_sscp_hcf_object_id;

  sycl::queue q = get_queue();
  int* data = sycl::malloc_device<int>(1,q);

  // This is just there to compile instantiate kernels, we don't actually want
  // to run them. So always_false should always evaluate to false 
  // without allowing the compiler to optimize it away.
  if(always_false) {
    q.single_task(__test_kernel1_{data, unannotated_wrapper<int>{}});
    q.single_task(__test_kernel2_{
        data,
        __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<int>{}});
    q.single_task(__test_kernel3_{
        data, __hipsycl_sscp_emit_param_type_annotation_specialized<int>{}});
    q.single_task(__test_kernel4_{
        data, __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
                  __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
                      int>>{}});
    q.single_task(__test_kernel5_{
        data,
        __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
            __hipsycl_sscp_emit_param_type_annotation_specialized<int>>{}});
    q.single_task(__test_kernel6_{
        data,
        __hipsycl_sscp_emit_param_type_annotation_custom_annotation1<
                     aggregate_type>{}});
  }

  q.wait();
  sycl::free(data, q);

  const hipsycl::rt::hcf_image_info *img_info =
      hipsycl::rt::hcf_cache::get().get_image_info(hcf_id, "llvm-ir.global");
  
  auto get_kernel_name = [&](const std::string& fragment) {
    for(const auto& candidate : img_info->get_contained_kernels())
      if(candidate.find(fragment) != std::string::npos)
        return candidate;
    return std::string{};
  };

  auto get_kernel_info = [&](const std::string &kernel_name_fragment) {
    return hipsycl::rt::hcf_cache::get().get_kernel_info(
        hcf_id, get_kernel_name(kernel_name_fragment));
  };

  auto kernel_info1 = get_kernel_info("__test_kernel1_");
  auto kernel_info2 = get_kernel_info("__test_kernel2_");
  auto kernel_info3 = get_kernel_info("__test_kernel3_");
  auto kernel_info4 = get_kernel_info("__test_kernel4_");
  auto kernel_info5 = get_kernel_info("__test_kernel5_");
  auto kernel_info6 = get_kernel_info("__test_kernel6_");

  // CHECK: 0
  // CHECK: 0
  // CHECK: 0
  // CHECK: 0
  std::cout << kernel_info1->get_string_annotations(0).size() << std::endl;
  std::cout << kernel_info1->get_string_annotations(1).size() << std::endl;
  std::cout << kernel_info1->get_known_annotations(0).size() << std::endl;
  std::cout << kernel_info1->get_known_annotations(1).size() << std::endl;

  // CHECK: 0
  // CHECK: 1
  // CHECK: 0
  // CHECK: 0
  std::cout << kernel_info2->get_string_annotations(0).size() << std::endl;
  std::cout << kernel_info2->get_string_annotations(1).size() << std::endl;
  std::cout << kernel_info2->get_known_annotations(0).size() << std::endl;
  std::cout << kernel_info2->get_known_annotations(1).size() << std::endl;
  // CHECK: custom_annotation1
  std::cout << kernel_info2->get_string_annotations(1)[0] << std::endl;
  
  // CHECK: 0
  // CHECK: 0
  // CHECK: 0
  // CHECK: 1
  std::cout << kernel_info3->get_string_annotations(0).size() << std::endl;
  std::cout << kernel_info3->get_string_annotations(1).size() << std::endl;
  std::cout << kernel_info3->get_known_annotations(0).size() << std::endl;
  std::cout << kernel_info3->get_known_annotations(1).size() << std::endl;
  // CHECK: 0
  std::cout << kernel_info3->get_known_annotations(1)[0] << std::endl;

  // CHECK: 0
  // CHECK: 1
  // CHECK: 0
  // CHECK: 0
  std::cout << kernel_info4->get_string_annotations(0).size() << std::endl;
  std::cout << kernel_info4->get_string_annotations(1).size() << std::endl;
  std::cout << kernel_info4->get_known_annotations(0).size() << std::endl;
  std::cout << kernel_info4->get_known_annotations(1).size() << std::endl;
  // CHECK: custom_annotation1
  std::cout << kernel_info4->get_string_annotations(1)[0] << std::endl;

  // CHECK: 0
  // CHECK: 1
  // CHECK: 0
  // CHECK: 1
  std::cout << kernel_info5->get_string_annotations(0).size() << std::endl;
  std::cout << kernel_info5->get_string_annotations(1).size() << std::endl;
  std::cout << kernel_info5->get_known_annotations(0).size() << std::endl;
  std::cout << kernel_info5->get_known_annotations(1).size() << std::endl;
  // CHECK: custom_annotation1
  std::cout << kernel_info5->get_string_annotations(1)[0] << std::endl;
  // CHECK: 0
  std::cout << kernel_info5->get_known_annotations(1)[0] << std::endl;

  // CHECK: 0
  // CHECK: 1
  // CHECK: 1
  // CHECK: custom_annotation1
  // CHECK: custom_annotation1
  std::cout << kernel_info6->get_string_annotations(0).size() << std::endl;
  std::cout << kernel_info6->get_string_annotations(1).size() << std::endl;
  std::cout << kernel_info6->get_string_annotations(2).size() << std::endl;
  std::cout << kernel_info5->get_string_annotations(1)[0] << std::endl;
  std::cout << kernel_info5->get_string_annotations(1)[0] << std::endl;
}
