/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "hipSYCL/sycl/buffer.hpp"
#include "hipSYCL/sycl/property.hpp"
#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/queue.hpp"

#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(extension_tests, reset_device_fixture)

#ifdef HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE
BOOST_AUTO_TEST_CASE(auto_placeholder_require_extension) {
  namespace s = cl::sycl;

  s::queue q;
  s::buffer<int, 1> buff{1};
  s::accessor<int, 1, s::access::mode::read_write, 
    s::access::target::global_buffer, s::access::placeholder::true_t> acc{buff};

  // This will call handler::require(acc) for each
  // subsequently launched command group
  auto automatic_requirement = s::vendor::hipsycl::automatic_require(q, acc);
  BOOST_CHECK(automatic_requirement.is_required());

  q.submit([&](s::handler &cgh) {
    cgh.single_task<class auto_require_kernel0>([=]() {
      acc[0] = 1;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 1);
  }

  q.submit([&] (s::handler& cgh) {
    cgh.single_task<class auto_require_kernel1>([=] (){
      acc[0] = 2;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 2);
  }

  automatic_requirement.release();
  BOOST_CHECK(!automatic_requirement.is_required());

  { 
    auto host_acc = buff.get_access<s::access::mode::read_write>(); 
    host_acc[0] = 3;
  }

  automatic_requirement.reacquire();
  BOOST_CHECK(automatic_requirement.is_required());

  q.submit([&] (s::handler& cgh) {
    cgh.single_task<class auto_require_kernel2>([=] (){
      acc[0] += 1;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 4);
  }
}
#endif
#ifdef HIPSYCL_EXT_CUSTOM_PFWI_SYNCHRONIZATION
BOOST_AUTO_TEST_CASE(custom_pfwi_synchronization_extension) {
  namespace sync = cl::sycl::vendor::hipsycl::synchronization;

  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](cl::sycl::handler& cgh) {

      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto scratch =
          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>{local_size,
                                                                    cgh};

      cgh.parallel_for_work_group<class pfwi_dispatch>(
        cl::sycl::range<1>{global_size / local_size},
        cl::sycl::range<1>{local_size},
        [=](cl::sycl::group<1> wg) {

          wg.parallel_for_work_item<sync::local_barrier>(
            [&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] = acc[item.get_global_id()];
          });

          // By default, a barrier is used
          wg.parallel_for_work_item(
            [&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] *= 2;
          });

          // Testing the behavior of mem_fence() or 
          // that there is no synchronization is difficult,
          // so let's just test that things compile for now.
          wg.parallel_for_work_item<sync::none>(
            [&](cl::sycl::h_item<1> item) {
            acc[item.get_global_id()] = scratch[item.get_local_id()[0]];
          });

          wg.parallel_for_work_item<sync::local_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });

          wg.parallel_for_work_item<sync::global_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });

          wg.parallel_for_work_item<sync::global_and_local_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });
        });
    });
  }

  for(size_t i = 0; i < global_size; ++i) {
    BOOST_TEST(host_buf[i] == 2*i);
  }
}
#endif
#ifdef HIPSYCL_EXT_SCOPED_PARALLELISM
BOOST_AUTO_TEST_CASE(scoped_parallelism_reduction) {
  namespace s = cl::sycl;
  s::queue q;
  
  std::size_t input_size = 256;
  std::vector<int> input(input_size);
  for(int i = 0; i < input.size(); ++i)
    input[i] = i;
  
  s::buffer<int> buff{input.data(), s::range<1>{input_size}};
  
  constexpr size_t Group_size = 64;
  
  q.submit([&](s::handler& cgh){
    auto data_accessor = buff.get_access<s::access::mode::read_write>(cgh);
    cgh.parallel<class Kernel>(s::range<1>{input_size / Group_size}, s::range<1>{Group_size}, 
    [=](s::group<1> grp, s::physical_item<1> phys_idx){
      s::local_memory<int [Group_size]> scratch{grp};
      s::private_memory<int> load{grp};
      
      grp.distribute_for([&](s::sub_group sg, s::logical_item<1> idx){
          load(idx) = data_accessor[idx.get_global_id(0)];
      });
      grp.distribute_for([&](s::sub_group sg, s::logical_item<1> idx){
          scratch[idx.get_local_id(0)] = load(idx);
      });

      for(int i = Group_size / 2; i > 0; i /= 2){
        grp.distribute_for([&](s::sub_group sg, s::logical_item<1> idx){
          size_t lid = idx.get_local_id(0);
          if(lid < i)
            scratch[lid] += scratch[lid+i];
        });
      }
      
      grp.single_item([&](){
        data_accessor[grp.get_id(0)*Group_size] = scratch[0];
      });
    });
  });
  
  auto host_acc = buff.get_access<s::access::mode::read>();
  
  for(int grp = 0; grp < input_size/Group_size; ++grp){
    int host_result = 0;
    for(int i = grp * Group_size; i < (grp+1) * Group_size; ++i)
      host_result += i;
    
    BOOST_TEST(host_result == host_acc[grp * Group_size]);
  }
}
#ifdef HIPSYCL_EXT_ENQUEUE_CUSTOM_OPERATION
BOOST_AUTO_TEST_CASE(custom_enqueue) {
  using namespace cl;

#ifdef HIPSYCL_PLATFORM_CUDA
  constexpr sycl::backend target_be = sycl::backend::cuda;
#elif defined(HIPSYCL_PLATFORM_HIP)
  constexpr sycl::backend target_be = sycl::backend::hip;
#elif defined(HIPSYCL_PLATFORM_SPIRV)
  constexpr sycl::backend target_be = sycl::backend::level_zero;
#else
  constexpr sycl::backend target_be = sycl::backend::omp;
#endif

  sycl::queue q;
  const std::size_t test_size = 1024;

  std::vector<int> initial_data(test_size, 14);
  std::vector<int> target_data(test_size);
  int* target_ptr = target_data.data();

  sycl::buffer<int, 1> buff{initial_data.data(), sycl::range<1>{test_size}};

  q.submit([&](sycl::handler &cgh) {
    auto acc = buff.get_access<sycl::access::mode::read>(cgh);

    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
      // All backends support obtaining native memory
      void *native_mem = h.get_native_mem<target_be>(acc);

      // OpenMP backend doesn't support extracting a native queue or device
#ifdef HIPSYCL_PLATFORM_CUDA
      auto stream = h.get_native_queue<target_be>();
      // dev is not really used, just test that this function call works for now
      sycl::backend_traits<target_be>::native_type<sycl::device> dev =
          h.get_native_device<target_be>();

      cudaMemcpyAsync(target_ptr, native_mem, test_size * sizeof(int),
                      cudaMemcpyDeviceToHost, stream);

#elif defined(HIPSYCL_PLATFORM_HIP)
      
      auto stream = h.get_native_queue<target_be>();
      // dev is not really used, just test that this function call works for now
      sycl::backend_traits<target_be>::native_type<sycl::device> dev =
          h.get_native_device<target_be>();

      hipMemcpyAsync(target_ptr, native_mem, test_size * sizeof(int),
                      hipMemcpyDeviceToHost, stream);
#endif
    });
  });

  q.wait();

  if constexpr (target_be == sycl::backend::cuda ||
                target_be == sycl::backend::hip) {
    for (std::size_t i = 0; i < test_size; ++i) {
      BOOST_TEST(initial_data[i] == target_data[i]);
    }
  }
}
#endif


#endif
#ifdef HIPSYCL_EXT_CG_PROPERTY_RETARGET
BOOST_AUTO_TEST_CASE(cg_property_retarget) {
  using namespace cl;

  auto all_devices = sycl::device::get_devices();

  std::vector<sycl::device> target_devices;
  for(const auto& dev : all_devices) {
    if (dev.hipSYCL_has_compiled_kernels() && dev.is_gpu()) {
      target_devices.push_back(dev);
    }
  }
  sycl::device host_device{sycl::detail::get_host_device()};

  if(target_devices.size() > 0) {
    sycl::queue q{target_devices[0], sycl::property_list{sycl::property::queue::in_order{}}};
    int* ptr = sycl::malloc_shared<int>(1, q);
    *ptr = 0;

    q.parallel_for<class retarget_gpu_kernel>(sycl::range{128}, 
      [=](sycl::id<1> idx){
      
      if(idx[0] == 0)
        ++ptr[0];
    });

    q.submit({sycl::property::command_group::hipSYCL_retarget{host_device}},
      [&](sycl::handler& cgh){
        cgh.single_task<class retarget_host_kernel>([=](){
          ++ptr[0];
        });  
      });

    q.wait();

    BOOST_TEST(ptr[0] == 2);

    sycl::free(ptr, q);
  }
}
#endif

#if defined(HIPSYCL_PLATFORM_CUDA) || \
    defined(HIPSYCL_PLATFORM_HIP) || \
    defined(HIPSYCL_PLATFORM_SPIRV)
HIPSYCL_KERNEL_TARGET
int get_total_group_size() {
#ifdef SYCL_DEVICE_ONLY
  return __hipsycl_lsize_x * __hipsycl_lsize_y * __hipsycl_lsize_z;
#else
  return 0;
#endif
}
#endif

#ifdef HIPSYCL_EXT_CG_PROPERTY_PREFER_GROUP_SIZE
BOOST_AUTO_TEST_CASE(cg_property_preferred_group_size) {
  using namespace cl;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  int* gsize = sycl::malloc_shared<int>(3, q);

  auto group_size1d = sycl::range{100};
  auto group_size2d = sycl::range{9,9};
  auto group_size3d = sycl::range{5,5,5};

#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__) ||                                 \
    defined(__HIPSYCL_ENABLE_HIP_TARGET__)
#define HIPLIKE_MODEL
#endif

  q.submit({sycl::property::command_group::hipSYCL_prefer_group_size{
               group_size1d}},
           [&](sycl::handler &cgh) {
             cgh.parallel_for<class property_preferred_group_size1>(
                 sycl::range{1000}, [=](sycl::id<1> idx) {
                   if (idx[0] == 0) {
#if defined(SYCL_DEVICE_ONLY) && defined(HIPLIKE_MODEL)
                     gsize[0] = get_total_group_size();
#else
                     gsize[0] = 1;
#endif
                   }
                 });
           });

  q.submit({sycl::property::command_group::hipSYCL_prefer_group_size{
               group_size2d}},
           [&](sycl::handler &cgh) {
             cgh.parallel_for<class property_preferred_group_size2>(
                 sycl::range{30,30}, [=](sycl::id<2> idx) {
                   if (idx[0] == 0 && idx[1] == 0) {
#if defined(SYCL_DEVICE_ONLY) && defined(HIPLIKE_MODEL)
                     gsize[1] = get_total_group_size();
#else
                     gsize[1] = 2;
#endif
                   }
                 });
           });


  q.submit({sycl::property::command_group::hipSYCL_prefer_group_size{
               group_size3d}},
           [&](sycl::handler &cgh) {
             cgh.parallel_for<class property_preferred_group_size3>(
                 sycl::range{10,10,10}, [=](sycl::id<3> idx) {
                   if (idx[0] == 0 && idx[1] == 0) {
#if defined(SYCL_DEVICE_ONLY) && defined(HIPLIKE_MODEL)
                     gsize[2] = get_total_group_size();
#else
                     gsize[2] = 3;
#endif
                   }
                 });
           });

  q.wait();

#ifdef HIPLIKE_MODEL
  BOOST_TEST(gsize[0] == group_size1d.size());
  BOOST_TEST(gsize[1] == group_size2d.size());
  BOOST_TEST(gsize[2] == group_size3d.size());
#else
  BOOST_TEST(gsize[0] == 1);
  BOOST_TEST(gsize[1] == 2);
  BOOST_TEST(gsize[2] == 3);
#endif

  sycl::free(gsize, q);
}
#endif

#ifdef HIPSYCL_EXT_CG_PROPERTY_PREFER_EXECUTION_LANE

BOOST_AUTO_TEST_CASE(cg_property_prefer_execution_lane) {

  cl::sycl::queue q;

  // Only compile testing for now
  for(std::size_t i = 0; i < 100; ++i) {
    q.submit(
        {cl::sycl::property::command_group::hipSYCL_prefer_execution_lane{i}},
        [&](cl::sycl::handler &cgh) {
          cgh.single_task<class prefer_execution_lane_test>([=]() {});
        });
  }
  q.wait();
}

#endif

#ifdef HIPSYCL_EXT_PREFETCH_HOST
BOOST_AUTO_TEST_CASE(prefetch_host) {
  using namespace cl;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  int *shared_mem = sycl::malloc_shared<int>(test_size, q);

  for (std::size_t i = 0; i < test_size; ++i)
    shared_mem[i] = i;

  q.parallel_for<class usm_prefetch_host_test_kernel>(
      sycl::range<1>{test_size},
      [=](sycl::id<1> idx) { shared_mem[idx.get(0)] += 1; });
  q.prefetch_host(shared_mem, test_size * sizeof(int));
  q.wait_and_throw();

  for (std::size_t i = 0; i < test_size; ++i)
    BOOST_TEST(shared_mem[i] == i + 1);

  sycl::free(shared_mem, q);
}
#endif
#ifdef HIPSYCL_EXT_BUFFER_USM_INTEROP
BOOST_AUTO_TEST_CASE(buffer_introspection) {
  using namespace cl;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};
  sycl::range size{1024};

  int* usm_ptr = nullptr;
  {
    sycl::buffer<int> buff{size};

    q.submit([&](sycl::handler& cgh){
      auto acc = buff.get_access<sycl::access::mode::discard_write>(cgh);
      // Force allocation of buffer on target device
      cgh.single_task([=](){});
    });

    q.wait();

    BOOST_TEST(buff.has_allocation(q.get_device()));
    usm_ptr = buff.get_pointer(q.get_device());
    BOOST_TEST(usm_ptr != nullptr);

    // Query information
    sycl::buffer_allocation::descriptor<int> alloc =
        buff.get_allocation(usm_ptr);
    BOOST_TEST(alloc.ptr == usm_ptr);
    BOOST_CHECK(alloc.dev == q.get_device());
    BOOST_TEST(alloc.is_owned == true);

    // This doesn't change anything as the allocation is already
    // owned because the buffer constructor was not provided a pointer.
    // Execute both variants to make sure both interfaces work.
    buff.own_allocation(usm_ptr);
    buff.own_allocation(q.get_device());
    alloc = buff.get_allocation(usm_ptr);
    BOOST_TEST(alloc.is_owned == true);

    // Disown allocation so that we can use it outside the
    // buffer scope
    buff.disown_allocation(usm_ptr);

    alloc = buff.get_allocation(usm_ptr);
    BOOST_TEST(alloc.is_owned == false);

    std::vector<int*> allocations;
    buff.for_each_allocation(
        [&](const sycl::buffer_allocation::descriptor<int> &a) {
          allocations.push_back(a.ptr);
        });

    BOOST_TEST(allocations.size() >= 1);
    bool found = false;
    for(std::size_t i = 0; i < allocations.size(); ++i) {
      if(allocations[i] == usm_ptr)
        found = true;
    }
    BOOST_TEST(found);
  }

  // Use extracted USM pointer directly
  std::vector<int> host_mem(size[0]);

  q.parallel_for(size, [usm_ptr](sycl::id<1> idx){
    usm_ptr[idx[0]] = idx[0];
  });
  q.memcpy(host_mem.data(), usm_ptr, sizeof(int)*size[0]);
  q.wait();

  for(std::size_t i = 0; i < host_mem.size(); ++i) {
    BOOST_CHECK(host_mem[i] == i);
  }

  sycl::free(usm_ptr, q);


}

BOOST_AUTO_TEST_CASE(buffers_over_usm_pointers) {
  using namespace cl;

  sycl::queue q;
  sycl::range size{1024};

  int* alloc1 = sycl::malloc_shared<int>(size.size(), q);
  int* alloc2 = sycl::malloc_shared<int>(size.size(), q);

  {
    sycl::buffer<int> b1{
        {sycl::buffer_allocation::empty_view(alloc1, q.get_device())}, size};

    BOOST_CHECK(b1.has_allocation(q.get_device()));
    BOOST_CHECK(b1.get_pointer(q.get_device()) == alloc1);
    b1.for_each_allocation([&](const auto& alloc){
      if(alloc.ptr == alloc1){
        BOOST_CHECK(!alloc.is_owned);
      }
    });

    q.submit([&](sycl::handler& cgh){
      sycl::accessor<int> acc{b1, cgh};

      cgh.parallel_for(size, [=](sycl::id<1> idx){
        acc[idx] = idx.get(0);
      });
    });
  }
  q.wait();
  for(int i = 0; i < size.get(0); ++i){
    BOOST_CHECK(alloc1[i] == i);
  }
  {
    sycl::buffer<int> b2{
        {sycl::buffer_allocation::view(alloc1, q.get_device())}, size};
    
    q.submit([&](sycl::handler& cgh){
      sycl::accessor<int> acc{b2, cgh};

      cgh.parallel_for(size, [=](sycl::id<1> idx){
        alloc2[idx.get(0)] = acc[idx];
      });
    });

    // Check that data state tracking works and migrating back to host
    sycl::host_accessor<int> hacc{b2};
    for(int i = 0; i < size.get(0); ++i){
      BOOST_CHECK(hacc[i] == i);
    }  
  }
  
  for(int i = 0; i < size.get(0); ++i){
    BOOST_CHECK(alloc2[i] == i);
  }

  sycl::free(alloc1, q);
  sycl::free(alloc2, q);
}

#endif
#ifdef HIPSYCL_EXT_BUFFER_PAGE_SIZE

BOOST_AUTO_TEST_CASE(buffer_page_size) {
  using namespace cl;

  sycl::queue q;

  // Deliberately choose page_size so that size is not a mulitple of it
  // to test the more complicated case.
  const std::size_t size = 1000;
  const std::size_t page_size = 512;
  sycl::buffer<int, 2> buff{sycl::range{size, size},
                            sycl::property::buffer::hipSYCL_page_size<2>{
                                sycl::range{page_size, page_size}}};

  // We have 4 pages
  for(std::size_t offset_x = 0; offset_x < size; offset_x += page_size) {
    for(std::size_t offset_y = 0; offset_y < size; offset_y += page_size) {
      auto event = q.submit([&](sycl::handler &cgh) {

        sycl::range range{std::min(page_size, size - offset_x),
                          std::min(page_size, size - offset_y)};
        sycl::id offset{offset_x, offset_y};

        for(int i = 0; i < 2; ++i){
          assert(offset[i]+range[i] <= size);
        }

        sycl::accessor<int, 2> acc{buff, cgh, range, offset};

        cgh.parallel_for(sycl::range{range}, [=](sycl::id<2> idx){
          // TODO this needs to be changed once we have SYCL 2020
          // semantics for operator[] of ranged accesors
          acc[idx + offset] =
              static_cast<int>(idx[0] + offset[0] + idx[1] + offset[1]);
        });
      });

      // All kernels should be independent, in that case we should
      // have a wait list of exactly one element: The one accessor
      // we have requested.
      // TODO This does not really guarantee that the kernels run in-
      // dependently as access conflicts are typically added to the requirements
      // of the accessor, not the kernel.
      BOOST_CHECK(event.get_wait_list().size() == 1);
    }
  }

  sycl::host_accessor<int, 2> hacc{buff};

  for(int i = 0; i < size; ++i) {
    for(int j = 0; j < size; ++j) {
      BOOST_REQUIRE(hacc[i][j] == i+j);
    }
  }
}

#endif
#ifdef HIPSYCL_EXT_EXPLICIT_BUFFER_POLICIES
BOOST_AUTO_TEST_CASE(explicit_buffer_policies) {
  using namespace cl;
  sycl::queue q;
  sycl::range size{1024};

  {
    std::vector<int> input_vec(size.size());
    
    for(int i = 0; i < input_vec.size(); ++i)
      input_vec[i] = i;
    
    auto b1 = sycl::make_async_buffer(input_vec.data(), size);
    // Because of buffer semantics we should be able to modify the input
    // pointer again
    input_vec[20] = 0;

    q.submit([&](sycl::handler& cgh){
      sycl::accessor acc{b1, cgh};
      cgh.parallel_for(size, [=](sycl::id<1> idx){
        acc[idx.get(0)] += 1;
      });
    });

    sycl::host_accessor hacc{b1};
    for(int i = 0; i < size.size(); ++i) {
      BOOST_CHECK(hacc[i] == i+1);
    }

    // Submit another operation before buffer goes out of
    // scope to make sure operations work even if the buffer leaves
    // scope.
    q.submit([&](sycl::handler& cgh){
      sycl::accessor acc{b1, cgh};
      cgh.parallel_for(size, [=](sycl::id<1> idx){
        acc[idx.get(0)] -= 1;
      });
    });
  }

  {
    std::vector<int> input_vec(size.size());
    
    for(int i = 0; i < input_vec.size(); ++i)
      input_vec[i] = i;
    {
      auto b1 = sycl::make_sync_writeback_view(input_vec.data(), size);

      q.submit([&](sycl::handler& cgh){
        sycl::accessor acc{b1, cgh};
        cgh.parallel_for(size, [=](sycl::id<1> idx){
          acc[idx.get(0)] += 1;
        });
      });
    }
    for(int i = 0; i < input_vec.size(); ++i) {
      BOOST_CHECK(input_vec[i] == i+1);
    }
  }

  {
    std::vector<int> input_vec(size.size());
    
    for(int i = 0; i < input_vec.size(); ++i)
      input_vec[i] = i;
    {
      auto b1 = sycl::make_async_writeback_view(input_vec.data(), size, q);

      q.submit([&](sycl::handler& cgh){
        sycl::accessor acc{b1, cgh};
        cgh.parallel_for(size, [=](sycl::id<1> idx){
          acc[idx.get(0)] += 1;
        });
      });
    }

    q.wait();

    for(int i = 0; i < input_vec.size(); ++i) {
      BOOST_CHECK(input_vec[i] == i+1);
    }
  }



}
#endif
#ifdef HIPSYCL_EXT_KERNEL_STATIC_PROPERTY_LIST
BOOST_AUTO_TEST_CASE(kernel_static_property_list) {
  using namespace cl;
  sycl::queue q;
  sycl::nd_range<1> rng{1024,128};

  bool exception_encountered = false;

  try {
    q.parallel_for(rng, sycl::attribute<sycl::reqd_work_group_size<128>>(
                            [=](sycl::nd_item<1> idx) {

                            }))
        .wait();
  } catch(...) {
    exception_encountered = true;
  }
  BOOST_CHECK(!exception_encountered);

  try {
    // Invalid work group size: rng requests 128 but the attribute requires 64
    // This should result in a synchronous exception.
    q.parallel_for(rng, sycl::attribute<sycl::reqd_work_group_size<64>>(
                            [=](sycl::nd_item<1> idx) {

                            }))
        .wait();
  } catch(...) {
    exception_encountered = true;
  }
  BOOST_CHECK(exception_encountered);
}

#endif
BOOST_AUTO_TEST_SUITE_END()
