// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s

//#define ENABLE_DOUBLE_LOOP

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main(int argc, char **argv) {
  sycl::device device(sycl::default_selector_v);
  sycl::queue queue(device);

  const int nrow = 2;
  const int ncol = 3;
  const int nsource = 16;

  int *d_data = static_cast<int *>(
      sycl::malloc_device(nrow * ncol * nsource * sizeof(int), queue));

  int *d_reduction =
      static_cast<int *>(sycl::malloc_device(nrow * ncol * sizeof(int), queue));

  std::vector<int> h_data(nrow * ncol * nsource);
  std::vector<int> h_reduction_correct(nrow * ncol);
  std::fill(h_reduction_correct.begin(), h_reduction_correct.end(), 0);

  int v = 0;
  for (int sx = 0; sx < nsource; sx++) {
    for (int rx = 0; rx < nrow; rx++) {
      for (int cx = 0; cx < ncol; cx++) {
        h_data.at(nsource * (cx + rx * ncol) + sx) = v;
        h_reduction_correct.at(cx + rx * ncol) += v;
        v++;
      }
    }
  }

      
  queue.fill(d_reduction, (int)0, nrow * ncol);
  queue.memcpy(d_data, h_data.data(), nrow * ncol * nsource * sizeof(int))
      .wait_and_throw();

  queue.submit([&](sycl::handler &cgh){
      sycl::local_accessor<int> local_acc(sycl::range<1>(nsource), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(nsource), sycl::range<1>(nsource)),
          [=](sycl::nd_item<1> idx) {
            const int local_sycl_range = idx.get_local_range(0);
            const unsigned int local_sycl_index = idx.get_local_id(0);
            const int half_sycl_range = local_sycl_range / 2;

#ifdef ENABLE_DOUBLE_LOOP
            for (int rx = 0; rx < nrow; rx++) {
              for (int cx = 0; cx < ncol; cx++) {
                const int ex = cx + rx * ncol;
#else
            for (int ex = 0; ex < ncol*nrow; ex++) {
#endif

                const int offset = ex * local_sycl_range;
                for (unsigned int s = half_sycl_range; s > 0; s >>= 1) {
                  if (local_sycl_index < s) {
                    d_data[local_sycl_index + offset] +=
                        d_data[local_sycl_index + offset + s];
                  }
                  sycl::group_barrier(idx.get_group());
                }
                if (local_sycl_index == 0) {
                  d_reduction[ex] = d_data[offset];
                }
#ifdef ENABLE_DOUBLE_LOOP
                // Extra barrier does not fix double loop case.
                // sycl::group_barrier(idx.get_group());
              }
            }
#else

              // This will fix the single loop case if uncommented.
            //   sycl::group_barrier(idx.get_group());
            }
#endif
          });
      })
      .wait_and_throw();

  std::vector<int> h_reduction_to_test(nrow * ncol);
  queue
      .memcpy(h_reduction_to_test.data(), d_reduction,
              nrow * ncol * sizeof(int))
      .wait_and_throw();

  bool fail = false;
  for (int ex = 0; ex < nrow * ncol && (!fail); ex++) {
    if (h_reduction_to_test.at(ex) != h_reduction_correct.at(ex)) {
      std::cout << "FAILED "<< ex << " " << h_reduction_to_test.at(ex) << " " << h_reduction_correct.at(ex) << std::endl;
      fail = true;
    }
  }
  if (!fail) {
    std::cout << "PASSED" << std::endl;
  }

  sycl::free(d_reduction, queue);
  sycl::free(d_data, queue);
}

// CHECK: PASSED
