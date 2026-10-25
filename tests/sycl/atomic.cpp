/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#include "sycl_test_suite.hpp"
#include <cstddef>
#include <limits>
#include <type_traits>
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#endif

BOOST_FIXTURE_TEST_SUITE(atomic_tests, reset_device_fixture)

template<class T, class IntT>
T int_to_t(IntT i) {
  T result{};

  if constexpr(std::is_pointer_v<T>) {
    result = reinterpret_cast<T>(i);
  } else {
    result = static_cast<T>(i);
  }

  return result;
}

template<class T>
unsigned long long t_to_int(T x) {
  unsigned long long result = 0ull;

  if constexpr(std::is_pointer_v<T>) {
    result = reinterpret_cast<unsigned long long>(x);
  } else {
    result = static_cast<unsigned long long>(x);
  }

  return result;
}

using exchange_test_types =
    boost::mp11::mp_list<int, unsigned int, long, unsigned long, long long,
                     unsigned long long, float, double, int *>;
// mainly compile test
BOOST_AUTO_TEST_CASE_TEMPLATE(load_store_exchange, Type,
                              exchange_test_types) {
  sycl::queue q;
  SKIP_IF_MOLTENVK(q.get_device())

  if constexpr(std::is_same_v<Type, double>) {
    if (!q.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  if constexpr(sizeof(Type) == 8) {
    if (!q.get_device().has(sycl::aspect::atomic64)) {
      BOOST_TEST_MESSAGE("Skipping test for 64-bit atomics since device has no atomic64 support");
      return;
    }
  }

  Type initial = int_to_t<Type>(0);
  sycl::buffer<Type> b{&initial, sycl::range{1}};

  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{b, cgh, sycl::read_write};
    cgh.single_task([=]() {
      sycl::atomic_ref<Type, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          r{acc[0]};
      unsigned long long x = t_to_int(r.load());

      // There seems to be a bug in clang related to incorrect
      // code generation at least on CUDA for floating point
      // stores.
      if constexpr(!std::is_floating_point_v<Type>)
        r.store(int_to_t<Type>(x+2));

      r.exchange(int_to_t<Type>(x+1));
    });
  });
  sycl::host_accessor hacc{b};

  BOOST_CHECK(t_to_int(hacc[0]) == 1);
}


template <class T, class AtomicOp, class Verifier>
void atomic_device_reduction_test(AtomicOp op, Verifier v,
                                  std::string_view type_name, std::string_view op_name,
                                  std::function<int(int)> init = [](int t) { return -t; }) {

  sycl::queue q;
  if constexpr(std::is_same_v<T, double>) {
    if (!q.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2125
  if (q.get_device().get_backend() == sycl::backend::vk) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      BOOST_TEST_MESSAGE(
          "Skipping test since Vulkan has not floating point atomic support");
      return;
    }
  }

  if constexpr(sizeof(T) == 8) {
    if (!q.get_device().has(sycl::aspect::atomic64)) {
      BOOST_TEST_MESSAGE("Skipping test for 64-bit atomics since device has no atomic64 support");
      return;
    }
  }

  const std::size_t size = 2048;
  sycl::buffer<T> b{sycl::range{size}};
  {
    sycl::host_accessor hacc{b};
    for(std::size_t i = 0; i < size; ++i) {
      hacc[i] = int_to_t<T>(init(i));
    }
  }
  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{b, cgh, sycl::read_write};
    cgh.parallel_for(sycl::range{size}, [=](sycl::id<1> idx){
      if(idx.get(0) != 0) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                        sycl::memory_scope::device> r{acc[0]};
        if constexpr(std::is_pointer_v<T>) {
          op(r, reinterpret_cast<std::ptrdiff_t>(acc[idx]));
        } else {
          op(r, acc[idx]);
        }
      }
    });
  });
  {
    sycl::host_accessor hacc{b};
    T expected = int_to_t<T>(init(0));
    for(std::size_t i = 1; i < size; ++i) {
      if constexpr(std::is_pointer_v<T>) {
        v(expected, static_cast<std::ptrdiff_t>(init(i)));
      } else {
        v(expected, int_to_t<T>(init(i)));
      }
    }
    BOOST_TEST_CONTEXT("Checking result for " << op_name << " on " << type_name << ": "
        << expected << " (expected) == " << hacc[0] << " (received)") {
      BOOST_CHECK(expected == hacc[0]);
    }
  }
}

template <class T, class AtomicOp, class Verifier>
void atomic_local_reduction_test(AtomicOp op, Verifier v,
                                std::string_view type_name,
                                std::string_view op_name,
                                T init_val, T per_item_val) {
  sycl::queue q;

  // https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2125
  if (q.get_device().get_backend() == sycl::backend::vk) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
      BOOST_TEST_MESSAGE(
          "Skipping test since Vulkan has not floating point atomic support");
      return;
    }
  }

  if constexpr (std::is_same_v<T, double>) {
    if (!q.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }
  if constexpr (sizeof(T) == 8) {
    if (!q.get_device().has(sycl::aspect::atomic64)) {
      BOOST_TEST_MESSAGE("Skipping test for 64-bit atomics since device has no atomic64 support");
      return;
    }
  }

  const std::size_t local_size = 64;
  const std::size_t num_groups = 8;
  const std::size_t global_size = local_size * num_groups;

  sycl::buffer<T> result_buf{sycl::range{num_groups}};

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor result{result_buf, cgh, sycl::read_write, sycl::no_init};
    sycl::local_accessor<T, 1> local_acc{sycl::range{1}, cgh};

    cgh.parallel_for(sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
      auto group = item.get_group();
      auto local_id = item.get_local_id(0);

      if (local_id == 0) {
        local_acc[0] = init_val;
      }
      sycl::group_barrier(group);

      {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>
        local_ref{local_acc[0]};

        op(local_ref, per_item_val);
      }

      sycl::group_barrier(group);

      if (local_id == 0) {
        result[item.get_group(0)] = local_acc[0];
      }
    });
  });

  sycl::host_accessor hacc{result_buf};
  T expected = init_val;
  for (std::size_t i = 0; i < local_size; ++i) {
    v(expected, per_item_val);
  }

  for (std::size_t g = 0; g < num_groups; ++g) {
    BOOST_TEST_CONTEXT("Checking local result for " << op_name << " on " << type_name
      << " (group " << g << "): " << expected << " (expected) == " << hacc[g] << " (received)") {
      BOOST_CHECK(expected == hacc[g]);
    }
  }
}

BOOST_AUTO_TEST_CASE(fetch_add_unsigned_int) {
    unsigned int data = 0u;

    ::sycl::atomic_ref<unsigned int,
                       ::sycl::memory_order::relaxed,
                       ::sycl::memory_scope::device,
                       ::sycl::access::address_space::global_space>
    ref(data);

    ref.fetch_add(1);
    BOOST_TEST(data  == 1u);

    ref.fetch_add(5);
    BOOST_TEST(data  == 6u);

    ref.fetch_add(10);
    BOOST_TEST_CONTEXT("Checking result for fetch_add() on unsigned integers: "
        << 16u << " (expected) == " << data << " (received)") {
      BOOST_CHECK(data == 16u);
    }
}

BOOST_AUTO_TEST_CASE(fetch_op) {
  SKIP_IF_MOLTENVK(sycl::device{})

  auto fetch_add = [](auto& atomic, auto x) {
    return atomic.fetch_add(x);
  };

  auto fetch_sub = [](auto& atomic, auto x) {
    return atomic.fetch_sub(x);
  };

  auto fetch_and = [](auto& atomic, auto x) {
    return atomic.fetch_and(x);
  };

  auto fetch_or = [](auto& atomic, auto x) {
    return atomic.fetch_or(x);
  };

  auto fetch_xor = [](auto& atomic, auto x) {
    return atomic.fetch_xor(x);
  };

  auto fetch_min = [](auto& atomic, auto x) {
    return atomic.fetch_min(x);
  };

  auto fetch_max = [](auto& atomic, auto x) {
    return atomic.fetch_max(x);
  };


  auto fetch_add_verifier = [](auto& v, auto x) {
    v += x;
  };

  auto fetch_sub_verifier = [](auto& v, auto x) {
    v -= x;
  };

  auto fetch_and_verifier = [](auto& v, auto x) {
    v &= x;
  };

  auto fetch_or_verifier = [](auto& v, auto x) {
    v |= x;
  };

  auto fetch_xor_verifier = [](auto& v, auto x) {
    v ^= x;
  };

  auto fetch_min_verifier = [](auto& v, auto x) {
    v = std::min(v,x);
  };

  auto fetch_max_verifier = [](auto& v, auto x) {
    v = std::max(v,x);
  };

#define HIPSYCL_ATOMIC_REF_TEST_T(Type, Op, Verifier)                              \
  atomic_device_reduction_test<Type>(Op, Verifier, #Type, #Op);

#define HIPSYCL_ATOMIC_REF_INTEGER_TEST(Op, Verifier)                              \
  HIPSYCL_ATOMIC_REF_TEST_T(int, Op, Verifier)                                     \
  HIPSYCL_ATOMIC_REF_TEST_T(unsigned int, Op, Verifier)                            \
  HIPSYCL_ATOMIC_REF_TEST_T(long, Op, Verifier)                                    \
  HIPSYCL_ATOMIC_REF_TEST_T(unsigned long, Op, Verifier)                           \
  HIPSYCL_ATOMIC_REF_TEST_T(long long, Op, Verifier)                               \
  HIPSYCL_ATOMIC_REF_TEST_T(unsigned long long, Op, Verifier)

#define HIPSYCL_ATOMIC_REF_FP_TEST(Op, Verifier)                                   \
  HIPSYCL_ATOMIC_REF_TEST_T(float, Op, Verifier);                                  \
  HIPSYCL_ATOMIC_REF_TEST_T(double, Op, Verifier);

#define HIPSYCL_ATOMIC_REF_PTR_TEST(Op, Verifier, Initializer)                     \
  atomic_device_reduction_test<int *>(Op, Verifier, "int*", #Op, Initializer);

#ifndef ACPP_LIBKERNEL_CUDA_NVCXX

  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_sub, fetch_sub_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_or, fetch_or_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_xor, fetch_xor_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_and, fetch_and_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_min, fetch_min_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_max, fetch_max_verifier);

  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_sub, fetch_sub_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_min, fetch_min_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_max, fetch_max_verifier);

  HIPSYCL_ATOMIC_REF_PTR_TEST(fetch_add, fetch_add_verifier, [](auto t) {
    return t+1;
  });

  HIPSYCL_ATOMIC_REF_PTR_TEST(fetch_sub, fetch_sub_verifier, [](auto t) {
    if (t == 0)
      return std::numeric_limits<int>::max();
    else
      return t;
  });

#endif
}


BOOST_AUTO_TEST_CASE(fetch_add_sub_local_memory) {
  auto fetch_add = [](auto &atomic, auto x) {
    return atomic.fetch_add(x);
  };
  auto fetch_sub = [](auto &atomic, auto x) {
    return atomic.fetch_sub(x);
  };

  auto add_verifier = [](auto &v, auto x) {
    v += x;
  };
  auto sub_verifier = [](auto &v, auto x) {
    v -= x;
  };

#define ACPP_LOCAL_ATOMIC_REF_TEST_T(type, op, init, per_item_val) \
  atomic_local_reduction_test<type>(fetch_ ## op, op ## _verifier, #type, "fetch_" #op, init, per_item_val)

  ACPP_LOCAL_ATOMIC_REF_TEST_T(float, add, 0.0f, 1.0f);
  ACPP_LOCAL_ATOMIC_REF_TEST_T(float, sub, 64.0f, 1.0f);
  ACPP_LOCAL_ATOMIC_REF_TEST_T(double, add, 0.0f, 1.0f);
  ACPP_LOCAL_ATOMIC_REF_TEST_T(double, sub, 64.0f, 1.0f);

  ACPP_LOCAL_ATOMIC_REF_TEST_T(int, add, 0, 1);
  ACPP_LOCAL_ATOMIC_REF_TEST_T(int, sub, 64, 1);
  ACPP_LOCAL_ATOMIC_REF_TEST_T(unsigned int, add, 0u, 1u);
  ACPP_LOCAL_ATOMIC_REF_TEST_T(unsigned int, sub, 64u, 1u);

#undef ACPP_LOCAL_ATOMIC_REF_TEST_T
}

#ifndef ACPP_LIBKERNEL_CUDA_NVCXX // nvc++ has some issue with this test
BOOST_AUTO_TEST_CASE(atomic_fence) {
  // This is mainly a compile-test. Testing atomic memory semantics is hard...

  sycl::queue q;
  SKIP_IF_MOLTENVK(q.get_device())

  int* data = sycl::malloc_device<int>(1, q);
  q.memset(data, 0, sizeof(int)).wait();
  size_t range = 1024;

  q.parallel_for(range, [=](auto idx){
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device> a{*data};

    ++a;
    sycl::atomic_fence(sycl::memory_order::relaxed, sycl::memory_scope::device);
  }).wait();

  int result;
  q.memcpy(&result, data, sizeof(int)).wait();

  BOOST_CHECK(result == range);

  sycl::free(data, q);
}
#endif

////---
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP

#endif
////---

namespace PTXPremitiveTests 
{
  static constexpr size_t WORK_ITEMS = 256;
  static constexpr size_t ITEMS_PER_GROUP = 32;
  static constexpr size_t WORK_GROUPS = 8;
  static constexpr size_t PER_TEST_ADDRESS_SPACE_COUNT = 2;
  static constexpr size_t MAX_TRIES = 8192;

  using Order = sycl::memory_order;
  using Scope = sycl::memory_scope;
  using Space = sycl::access::address_space;

  struct OrderPair
  {
    Order Success;
    Order Failure;
  };

  template<class T, Scope SC, Space SP>
  int exercise(T* ptr)
  {
    static constexpr T FirstStoreVal = static_cast<T>(19);
    static constexpr T InvalidExpected = static_cast<T>(18);
    static constexpr T SecondDesired = static_cast<T>(16);
    static constexpr  OrderPair TestPairs[] =
    {
      {Order::relaxed, Order::relaxed},
      {Order::acquire, Order::relaxed}, 
      {Order::acquire, Order::acquire},
      {Order::release, Order::relaxed},
      {Order::acq_rel, Order::relaxed}, 
      {Order::acq_rel, Order::acquire},
      {Order::seq_cst, Order::relaxed}, 
      {Order::seq_cst, Order::acquire},
      {Order::seq_cst, Order::seq_cst}
    }; 

    sycl::atomic_ref<T, Order::relaxed, SC, SP> atom{*ptr};
    int errors = 0;

    const Order loads[] = {Order::relaxed, Order::acquire, Order::seq_cst};
    const Order stores[] = {Order::relaxed, Order::release, Order::seq_cst};

    for (Order st : stores) 
    {
      atom.store(FirstStoreVal, st);
      for (Order ld : loads) 
      {
        errors += atom.load(ld) != FirstStoreVal;
      }
    }

    for (OrderPair pr : TestPairs) 
    {
      atom.store(FirstStoreVal, Order::relaxed); // stored 18
      T expected = InvalidExpected;

      errors += atom.compare_exchange_strong(expected, SecondDesired, pr.Success, pr.Failure); // cmpx cant success because we are expecting 19 and value is 18
      errors += expected != FirstStoreVal; // returned expected should be our stored value 19

      expected = atom.load(Order::relaxed);

      errors += expected != FirstStoreVal; // load should show the same
      errors += !atom.compare_exchange_strong(expected, SecondDesired, pr.Success, pr.Failure); // now a valid CMPX
      errors += expected != FirstStoreVal;
      errors += atom.load(Order::relaxed) != SecondDesired; // value should be 16 now

      expected = InvalidExpected;

      errors += atom.compare_exchange_weak(expected, SecondDesired, pr.Success, pr.Failure); //Under the hood failure uses strong and it should fail because expected is 16
      errors += expected != SecondDesired; // returned expected should be 16

    }
    return errors;
  } 

  template<class T, Scope SC>
  void global_matrix(sycl::queue queue)
  {
      sycl::buffer<T> storage{sycl::range<1>{1}};
      int errors[PER_TEST_ADDRESS_SPACE_COUNT] = {};
      {
          sycl::buffer<int> result{errors, sycl::range<1>{PER_TEST_ADDRESS_SPACE_COUNT}};
          queue.submit([&](sycl::handler& hndl)
          {
              auto data = storage.template
                  get_access<sycl::access::mode::read_write>(hndl);

              auto out = result.get_access<sycl::access::mode::write>(hndl);

              hndl.single_task([=]()
              {
                  out[0] = exercise<T, SC, Space::global_space>(&data[0]);
                  out[1] = exercise<T, SC, Space::generic_space>(&data[0]);
              });
          });
          queue.wait_and_throw();
      }
      BOOST_CHECK_EQUAL(errors[0], 0);
      BOOST_CHECK_EQUAL(errors[1], 0);
  }

  template<class T>
  void local_matrix(sycl::queue& queue)
  {
      int errors[PER_TEST_ADDRESS_SPACE_COUNT] = {};
      {
        sycl::buffer<int> result{errors, sycl::range<1>{PER_TEST_ADDRESS_SPACE_COUNT}};
        queue.submit([&](sycl::handler& hndl)
        {
          sycl::local_accessor<T, 1> data{sycl::range<1>{1}, hndl}; //one element
          auto out = result.get_access<sycl::access::mode::write>(hndl);

          hndl.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{ITEMS_PER_GROUP}, sycl::range<1>{ITEMS_PER_GROUP}},
            [=](sycl::nd_item<1> item)
            {
              if(item.get_local_linear_id() == 0)
              {
                out[0] = exercise<T, Scope::work_group, Space::local_space>(&data[0]);
                out[1] = exercise<T, Scope::sub_group, Space::generic_space>(&data[0]);
              }
            }
          );
        });
        queue.wait_and_throw();
      }

      BOOST_CHECK_EQUAL(errors[0], 0);
      BOOST_CHECK_EQUAL(errors[1], 0);
  }

  template<class T>
  void contention(sycl::queue& queue)
  {
    T value = 0;
    int errors[WORK_ITEMS] = {};
    {
      sycl::buffer<T> data{&value, sycl::range<1>{1}};
      sycl::buffer<int> result{errors, sycl::range<1>{WORK_ITEMS}};

      queue.submit(
        [&](sycl::handler& hndl)
        {
          auto storage = data.template get_access<sycl::access::mode::read_write>(hndl);
          auto out = result.get_access<sycl::access::mode::write>(hndl);
          hndl.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{WORK_ITEMS}, sycl::range<1>{ITEMS_PER_GROUP}},
            [=](sycl::nd_item<1> item)
            {
              sycl::atomic_ref<T, Order::relaxed, Scope::device, Space::global_space> atom{storage[0]};

              int failures = 0;
              for (int i = 0; i < WORK_GROUPS; i++) 
              {
                T expected = atom.load(Order::relaxed);

                bool done = false;
                for (int r = 0; r < MAX_TRIES && !done; r++) 
                {
                  T prior = expected;
                  done = atom.compare_exchange_strong(expected, T(prior + 1), Order::acq_rel, Order::acquire);
                  failures += !done && expected <= prior; //suspicious failed CAS
                }
                failures += !done;
              }
              out[item.get_global_linear_id()] = failures;
            }
          );

        }
      );
      queue.wait_and_throw();
    }
    
    BOOST_CHECK_EQUAL(value, T(WORK_ITEMS * WORK_GROUPS));
    for (int e : errors)
    {
      BOOST_CHECK_EQUAL(e, 0);
    }
  }

  template<class T, bool FailedCAS>
  void publication(sycl::queue& queue)
  {
    static constexpr size_t PAIRS = 128;
    static constexpr int RANDOM_VAL = 123611;
    int observ = 0;
    for (int r = 0; r < WORK_GROUPS ; r++) 
    {
      T flags[PAIRS] = {};
      int payload[PAIRS] = {};
      int outcome[PAIRS] = {};
      {
        sycl::buffer<T> f_buf{flags, sycl::range<1>{PAIRS}};
        sycl::buffer<int> p_buf{payload, sycl::range<1>{PAIRS}};
        sycl::buffer<int> o_buf{outcome, sycl::range<1>{PAIRS}};

        queue.submit(
          [&](sycl::handler& hndl)
          {
            auto flag = f_buf.template get_access<sycl::access::mode::read_write>(hndl);
            auto data = p_buf.get_access<sycl::access::mode::read_write>(hndl);
            auto out = o_buf.get_access<sycl::access::mode::read_write>(hndl);

            hndl.parallel_for(
              sycl::nd_range<1>{sycl::range<1>{PAIRS * 64}, sycl::range<1>{ITEMS_PER_GROUP}},
              [=](sycl::nd_item<1> item)
              {
                if (item.get_global_linear_id() != 0) return;
                const auto group = item.get_global_linear_id();
                const auto pair = group / 2;
                const bool producer = (group % 2) == (r % 2);
                sycl::atomic_ref<T, Order::relaxed, Scope::device, Space::global_space> atom{flag[pair]};

                if (producer) 
                {
                  data[pair] = RANDOM_VAL;
                  atom.store(T{1}, Order::release);
                }
                else 
                {
                  T observed = T{0};

                  for (int atmp = 0; atmp < MAX_TRIES; atmp++) 
                  {
                    if constexpr (FailedCAS)
                    {
                      observed = T{2};
                      atom.compare_exchange_strong(observed, T{3}, Order::acquire, Order::acquire);
                    }
                    else 
                    {
                      observed = atom.load(Order::acquire);
                    }

                    if (observed == T{1}) break;
                  }
                  if (observed == T{1}) 
                  {
                    out[pair] = data[pair] == RANDOM_VAL ? 1 : -1;
                  }
                }
              }
            );
            //
          }
        );
        queue.wait_and_throw();
      }

      for (int r : outcome) 
      {
        BOOST_CHECK_MESSAGE(r != -1, "publication observed stale payload");
        observ += r == 1;
      }

    }
    BOOST_REQUIRE_MESSAGE(observ > 0,"Inconclusive scheduling: no publication observed; rerun this test");
    BOOST_TEST_MESSAGE("PTX publication observations: " << observ);
  }


  inline bool select_cuda(sycl::queue& queue);

}

BOOST_AUTO_TEST_SUITE_END()
