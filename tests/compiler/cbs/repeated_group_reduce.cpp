// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s

// this tests mostly -O0 for omp where there's a lot more alloca pain
// specifically this might trigger, if we don't arrayify allocas whose address
// is stored at another location and is thus used across subcfgs (without being
// unique / workitem)

#include <sycl/sycl.hpp>

using f64_3 = sycl::vec<double, 3>;

template <class T, class Func> inline T map_vector(const T &in, Func &&f) {
  return {f(in[0]), f(in[1]), f(in[2])};
}

int main() {
  sycl::queue q{};

  std::vector<f64_3> input_data = {
      {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
  std::vector<f64_3> output_data(input_data.size(), {0.0, 0.0, 0.0});
  {
    sycl::buffer<f64_3> input_buf(input_data.data(), input_data.size());
    sycl::buffer<f64_3> output_buf(output_data.data(), output_data.size());

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor in(input_buf, cgh, sycl::read_only);
      sycl::accessor out(output_buf, cgh, sycl::write_only);

      cgh.parallel_for(sycl::nd_range<1>(input_data.size(), input_data.size()),
                       [=](sycl::nd_item<1> item) {
                         sycl::group<1> group = item.get_group();

                         auto v = in[item.get_global_linear_id()];

                         f64_3 sum = map_vector(v, [&](auto component) {
                           auto ret = sycl::reduce_over_group(
                               group, component,
                               sycl::plus<decltype(component)>{});
                           return ret;
                         });

                         out[item.get_global_linear_id()] = sum;
                       });
    });
  }
  for (size_t i = 0; i < output_data.size(); ++i) {

    for (size_t j = 0; j < 3; ++j) {
      // CHECK: 22 26 30
      // CHECK: 22 26 30
      // CHECK: 22 26 30
      // CHECK: 22 26 30
      std::cout << output_data[i][j] << " ";
    }
    std::cout << "\n";
  }
}
