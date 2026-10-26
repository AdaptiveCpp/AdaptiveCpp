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

namespace AtomicOrderScopeTests 
{
  static constexpr size_t WORK_ITEMS = 256;
  static constexpr size_t ITEMS_PER_GROUP = 32;
  static constexpr size_t INCREMENT_PER_ITEM = 8;
  static_assert(WORK_ITEMS % ITEMS_PER_GROUP == 0, "total; items are not modulo of per item increment");
  static constexpr size_t PAIRS = 128;

  static constexpr size_t GROUPS_PER_PAIR = 2;
  static constexpr size_t PER_TEST_ADDRESS_SPACE_COUNT = 2;
  static const sycl::range<1> ONE_ELEMENT = sycl::range<1>{1};

  using Order = sycl::memory_order;
  using Scope = sycl::memory_scope;
  using Space = sycl::access::address_space;
  using IntTypes = boost::mp11::mp_list<unsigned int, long long>;
  using FloatTypes = boost::mp11::mp_list<float, double>;

  // More Tests can be added as we progress
  enum class PublicationTestMode : uint8_t
  {
    ACQUIRE_LOAD = 0,
    FAILED_CAS = 1,
    FLOATING_MIN_MAX = 2
  };

  template<class T>
  bool supports_type(const sycl::device& device)
  {
    if (sizeof(T) > sizeof(uint64_t)) 
    {
      BOOST_TEST_MESSAGE("Skipping : arbeterary sized atomics are not possible");
      return false;
    }

    if constexpr (sizeof(T) == sizeof(uint64_t)) 
    {
      if (!device.has(sycl::aspect::atomic64)) 
      {
        BOOST_TEST_MESSAGE("Skipping 64-bit atomic test: atomic64 unavailable");
        return false;
      }
    }

    if constexpr (std::is_same_v<T, double>)
    {
      if (!device.has(sycl::aspect::fp64)) 
      {
        BOOST_TEST_MESSAGE("Skipping double atomic test: fp64 unavailable");
        return false;
      }
    }
    return true;
  }

  template<class T, Scope SC, Space SP>
  int exercise_atomic_rmw(T* ptr)
  {
    static constexpr T FirstStoreVal = static_cast<T>(19);
    static constexpr T InvalidExpected = static_cast<T>(18);
    static constexpr T SecondDesired = static_cast<T>(16);

    struct OrderPair
    {
      Order Success;
      Order Failure;
    };
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
      atom.store(FirstStoreVal, Order::relaxed); // stored 19
      T expected = InvalidExpected;

      errors += atom.compare_exchange_strong(expected, SecondDesired, pr.Success, pr.Failure); // CAS fails: expected is 18, but atomic contains 19 and should update the expected for sync
      errors += expected != FirstStoreVal; // returned expected should be our stored value 19

      expected = atom.load(Order::relaxed);

      errors += expected != FirstStoreVal; // load should show the same
      errors += !atom.compare_exchange_strong(expected, SecondDesired, pr.Success, pr.Failure); // now a valid CMPX
      errors += expected != FirstStoreVal;
      errors += atom.load(Order::relaxed) != SecondDesired; // value should be 16 now

      expected = InvalidExpected;

      errors += atom.compare_exchange_weak(expected, SecondDesired, pr.Success, pr.Failure); //Under the hood cas_weak may be implimentation of strong Weak CAS must fail because expected != atomic value.
      errors += expected != SecondDesired; // returned expected should be 16

    }
    return errors;
  } 

  /// @brief NO Contension; Just verify single load, store, CAS - 2 calls 
  ///
  /// i) global_space , ii) generic_space calls same memory (stress compiler)
  template<class T, Scope SCOPE>
  void global_matrix(sycl::queue& queue)
  {
    sycl::buffer<T> storage{ONE_ELEMENT}; // one element storage on host
    int errors[PER_TEST_ADDRESS_SPACE_COUNT] = {};
    {
        sycl::buffer<int> result{errors, sycl::range<1>{PER_TEST_ADDRESS_SPACE_COUNT}};
        queue.submit([&](sycl::handler& hndl)
        {
            auto data = storage.template get_access<sycl::access::mode::read_write>(hndl);
            auto out = result.get_access<sycl::access::mode::write>(hndl);

            hndl.single_task([=]()
            {
                out[0] = exercise_atomic_rmw<T, SCOPE, Space::global_space>(&data[0]);
                out[1] = exercise_atomic_rmw<T, SCOPE, Space::generic_space>(&data[0]);
            });
        });
        queue.wait_and_throw();
    }
    BOOST_CHECK_EQUAL(errors[0], 0);
    BOOST_CHECK_EQUAL(errors[1], 0);
  }

  /// Reviewer can remove the test if unnecessery other tests also uses local memory
  /// @brief NO Contension; Just verify single load, store, CAS - 2 calls 
  ///
  /// i) local_space , ii) generic_space calls same memory (stress compiler)
  template<class T>
  void local_matrix(sycl::queue& queue)
  {
    int errors[PER_TEST_ADDRESS_SPACE_COUNT] = {};
    {
      sycl::buffer<int> result{errors, sycl::range<1>{PER_TEST_ADDRESS_SPACE_COUNT}};
      queue.submit([&](sycl::handler& hndl)
      {
        sycl::local_accessor<T, 1> data{ONE_ELEMENT, hndl}; //one element for kernel
        auto out = result.get_access<sycl::access::mode::write>(hndl);

        hndl.parallel_for(
          sycl::nd_range<1>{sycl::range<1>{ITEMS_PER_GROUP}, sycl::range<1>{ITEMS_PER_GROUP}}, //32 items
          [=](sycl::nd_item<1> item)
          {
            if(item.get_local_linear_id() == 0)
            {
              out[0] = exercise_atomic_rmw<T, Scope::work_group, Space::local_space>(&data[0]);
              out[1] = exercise_atomic_rmw<T, Scope::sub_group, Space::generic_space>(&data[0]);
            }
          }
        );
      });
      queue.wait_and_throw();
    }

    BOOST_CHECK_EQUAL(errors[0], 0);
    BOOST_CHECK_EQUAL(errors[1], 0);
  }

  /// real contenntion 256 worker each must execute 8 CAS
  template<class T>
  void contention(sycl::queue& queue)
  {
    static constexpr size_t CAS_RETRY_LIMIT = 4096;
    T value = 0;
    int errors[WORK_ITEMS] = {};
    {
      sycl::buffer<T> data{&value, ONE_ELEMENT};
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
              for (int i = 0; i < INCREMENT_PER_ITEM; i++) 
              {
                T expected = atom.load(Order::relaxed);

                bool done = false;
                for (int r = 0; r < CAS_RETRY_LIMIT && !done; r++) 
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
    
    BOOST_CHECK_EQUAL(value, T(WORK_ITEMS * INCREMENT_PER_ITEM));
    for (int e : errors)
    {
      BOOST_CHECK_EQUAL(e, 0);
    }
  }

  // Later based on demand we can convert publication into a actual class
  template<class T, PublicationTestMode M>
  void publication(sycl::queue& queue)
  {
    //observation enums
    static constexpr int DIDNT_OBSERVED = 0;
    static constexpr int PUBLICATION_OBSERVED_CORRECT_READ = 1;
    static constexpr int PUBLICATION_OBSERVED_INCORRECT_READ = -1;
    //info
    static constexpr int RANDOM_VAL = 123611;
    static constexpr size_t PUBLICATION_POOL_LIMIT = WORK_ITEMS * ITEMS_PER_GROUP;
    static constexpr size_t PUBLICATION_ROUNDS = 8;
    //flags
    static constexpr T FLAG_START = static_cast<T>(0);
    static constexpr T FLAG_READ_VALID = static_cast<T>(1);
    static constexpr T INVALID_OBSERVED = static_cast<T>(2);

    int observ = DIDNT_OBSERVED;
    for (int r = 0; r < PUBLICATION_ROUNDS ; r++) 
    {
      T flags[PAIRS] = {};

      //when we want to test FetchMinMax we pre defined to INVALID_OBSERVED so that fetch_min is valid later
      if constexpr (M == PublicationTestMode::FLOATING_MIN_MAX)
      {
        for (T& f : flags) 
        {
          f = INVALID_OBSERVED;
        }
      }

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
              sycl::nd_range<1>{sycl::range<1>{PAIRS * GROUPS_PER_PAIR * ITEMS_PER_GROUP}, sycl::range<1>{ITEMS_PER_GROUP}},
              [=](sycl::nd_item<1> item)
              {
                if (item.get_local_linear_id() != 0) return;
                const auto group = item.get_group_linear_id();
                const auto pair = group / GROUPS_PER_PAIR;
                const bool producer = (group % GROUPS_PER_PAIR) == (r % GROUPS_PER_PAIR);
                sycl::atomic_ref<T, Order::relaxed, Scope::device, Space::global_space> atom{flag[pair]};

                if (producer) 
                {
                  data[pair] = RANDOM_VAL;

                  //now if FetchMinMax is enabled we fetch_min is valid due to earlier predefination
                  if constexpr (M == PublicationTestMode::FLOATING_MIN_MAX) 
                  {
                    atom.fetch_min(FLAG_READ_VALID, Order::release);
                  }
                  else 
                  {
                    atom.store(FLAG_READ_VALID, Order::release);
                  }
                }
                else 
                {
                  T observed = FLAG_START;
                  for (int atmp = 0; atmp < PUBLICATION_POOL_LIMIT; atmp++) 
                  {
                    if constexpr (M == PublicationTestMode::FAILED_CAS)
                    {
                      observed = INVALID_OBSERVED;
                      atom.compare_exchange_strong(observed, T{3}, Order::acquire, Order::acquire); // invalid cmpx
                    }
                    else if constexpr (M == PublicationTestMode::FLOATING_MIN_MAX) 
                    {
                      observed = atom.fetch_max(FLAG_READ_VALID, Order::acquire); // leaves the value unchanged
                    }
                    else 
                    {
                      observed = atom.load(Order::acquire);
                    }

                    if (observed == FLAG_READ_VALID) break;
                  }
                  if (observed == FLAG_READ_VALID) 
                  {
                    out[pair] = data[pair] == RANDOM_VAL ?
                      PUBLICATION_OBSERVED_CORRECT_READ : PUBLICATION_OBSERVED_INCORRECT_READ;
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
        BOOST_CHECK_MESSAGE(r != PUBLICATION_OBSERVED_INCORRECT_READ, "publication observed stale payload");
        observ += r == PUBLICATION_OBSERVED_CORRECT_READ;
      }

    }
    BOOST_REQUIRE_MESSAGE(observ > DIDNT_OBSERVED,"Inconclusive scheduling: no publication observed; rerun this test");
    BOOST_TEST_MESSAGE("PTX publication observations: " << observ);
  }


  template<class T, bool FailedCAS>
  void sequential_consistancy(sycl::queue& queue)
  {
    static constexpr T FLAG_READ_VALID = static_cast<T>(1);
    static constexpr T INVALID_OBSERVED = static_cast<T>(2);

    T values[PAIRS * GROUPS_PER_PAIR] = {};
    int observations[PAIRS * GROUPS_PER_PAIR] = {};
    {
      sycl::buffer<T> data{values, sycl::range<1>{PAIRS * GROUPS_PER_PAIR}};
      sycl::buffer<int> result{observations, sycl::range<1>{PAIRS * GROUPS_PER_PAIR}};
      queue.submit(
        [&](sycl::handler& hndl)
        {
          auto storage = data.template get_access<sycl::access::mode::read_write>(hndl);
          auto out = result.get_access<sycl::access::mode::write>(hndl);
          hndl.parallel_for(
            sycl::nd_range<1>{
              sycl::range<1>{PAIRS * GROUPS_PER_PAIR * ITEMS_PER_GROUP}, 
              sycl::range<1>{ITEMS_PER_GROUP}
            },
            [=](sycl::nd_item<1> item)
            {
              if (item.get_local_linear_id() != 0) return;
              const auto self = item.get_group_linear_id();
              const auto other = self ^ 1;

              sycl::atomic_ref<T, Order::relaxed, Scope::device, Space::global_space> own_st{storage[self]};
              sycl::atomic_ref<T, Order::relaxed, Scope::device, Space::global_space> peer{storage[other]};

              own_st.store(FLAG_READ_VALID, Order::seq_cst);
              T observed;
              if constexpr (FailedCAS)
              {
                observed = INVALID_OBSERVED;
                peer.compare_exchange_strong(observed, T{3}, Order::seq_cst, Order::seq_cst);
              }
              else 
              {
                observed = peer.load(Order::seq_cst);
              }
              out[self] = static_cast<int>(observed);
            }
          );
          //
        }
      );
      //
      queue.wait_and_throw();
    }
    for (size_t p = 0; p < PAIRS; p++) 
    {
      BOOST_CHECK_MESSAGE(
        observations[GROUPS_PER_PAIR * p] != 0 || observations[GROUPS_PER_PAIR * p + 1] != 0, 
        "seq_cst store buffering forbidden outcome"
      );
    }
  }

  //PR-#2280 check_floating_fetch_min_max(host) can be removed eventually by this
  template<class T>
  void fetch_min_max_check(sycl::queue& queue)
  {
    static constexpr T FIRST_VAL = static_cast<T>(10);
    static constexpr T VALID_FETCH_MIN_VAL = static_cast<T>(5);
    static constexpr T VALID_FETCH_MAX_VAL = static_cast<T>(15);

    T init = 0;
    int errors = 0;
    {
      sycl::buffer<T> data{&init, ONE_ELEMENT};
      sycl::buffer<int> result{&errors,ONE_ELEMENT};
      queue.submit(
        [&](sycl::handler& hndl)
        {
          auto storage = data.template get_access<sycl:: access::mode::read_write>(hndl);
          auto out = result.get_access<sycl::access::mode::write>(hndl);
          hndl.single_task([=](){
            sycl::atomic_ref<T, Order::relaxed, Scope::device, Space::global_space> atom{storage[0]};
            const Order order_list [] = {
              Order::relaxed, Order::acquire, Order::release, 
              Order::acq_rel, Order::seq_cst
            };

            int count = 0;
            for (Order order : order_list) 
            {
              atom.store(FIRST_VAL, Order::relaxed);
              count += atom.fetch_min(VALID_FETCH_MIN_VAL, order) != FIRST_VAL; //after valid fetch_min returns previous val
              count += atom.load(Order::relaxed) != VALID_FETCH_MIN_VAL; // should return the fatch_min val 
              count += atom.fetch_min((VALID_FETCH_MIN_VAL + 2), order) != VALID_FETCH_MIN_VAL; //operation itself dosent modify the result (VALID_FETCH_MIN_VAL + 2)> VALID_FETCH_MIN_VAL
              count += atom.load(Order::relaxed) != VALID_FETCH_MIN_VAL; // still unchanged
              count += atom.fetch_max((VALID_FETCH_MIN_VAL - 2), order) != VALID_FETCH_MIN_VAL; //still fetch_max dosent modify 
              count += atom.fetch_max(VALID_FETCH_MAX_VAL, order) != VALID_FETCH_MIN_VAL;
              count += atom.fetch_min(FIRST_VAL, order) != VALID_FETCH_MAX_VAL; // now it returns previous fetch_max val and stors FIRST_VAL 
              count += atom.load(Order::relaxed) != FIRST_VAL;
            }
            out[0] = count;
          });
          //
        }
      );
      //
      queue.wait_and_throw();
    }
    BOOST_CHECK_EQUAL(errors, 0);
  }
}
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP
  namespace AtomicOrderScopeTests::CudaPTXTestSpace 
  {
    inline bool select_cuda(sycl::queue& queue)
    {
      const bool cuda = queue.get_device().get_backend() == sycl::backend::cuda;
      const char* ENV_NAME = "PTX_ATOMIC_CUDA_ORDER";

      if (std::getenv(ENV_NAME)) 
      {
        BOOST_REQUIRE_MESSAGE(cuda, "PTX atomic order validation requires an actual CUDA device");
      }
      if (!cuda) 
      {
        BOOST_TEST_MESSAGE("PTX-specific runtime test skipped: selected device is not CUDA");
        return false;
      }

      int target[2] = {};
      {
        sycl::buffer<int> prob{target, sycl::range<1>{2}};
        queue.submit(
          [&](sycl::handler& hndl)
          {
            auto out = prob.get_access<sycl::access::mode::write>(hndl);
            hndl.single_task([=](){
              __acpp_if_target_sscp(
                out[0] = __acpp_sscp_jit_reflect_target_arch();
                out[1] = static_cast<int>(__acpp_sscp_jit_reflect_compiler_backend());
              );
            });
            //
          }
        );
        //
        queue.wait_and_throw();
      }
      BOOST_REQUIRE_MESSAGE(
        target[1] == static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::ptx),
        "SSCP PTX PATH REQUIRED"
      );

      if (std::getenv(ENV_NAME)) 
      {
        BOOST_REQUIRE_MESSAGE(target[0] >= 70, "cuda validation requirs SM70+");
      }

      if (target[0] < 70) 
      {
        BOOST_TEST_MESSAGE("PTX tests skipped: pre-SM70 legacy path is still undefined :(");
        return false;
      }

      BOOST_TEST_MESSAGE(
        "Executing SSCP PTX tests on " <<
        queue.get_device().get_info<sycl::info::device::name>() << 
        " SM " << target[0]
      );
      return true;
    }

    // TEST 01 -PTX
    BOOST_AUTO_TEST_CASE_TEMPLATE(ptx_premitive_matrix, T, IntTypes)
    {
      sycl::queue queue;
      if (!select_cuda(queue) || !supports_type<T>(queue.get_device())) return;

      global_matrix<T, Scope::sub_group>(queue);
      global_matrix<T, Scope::work_group>(queue);
      global_matrix<T, Scope::device>(queue);
      global_matrix<T, Scope::system>(queue); // This dosent test CPU <-> GPU sync

      local_matrix<T>(queue);
    }

    //TEST 02 
    BOOST_AUTO_TEST_CASE_TEMPLATE(ptx_publication, T, IntTypes)
    {
      sycl::queue queue;
      if (!select_cuda(queue) || !supports_type<T>(queue.get_device())) return;

      publication<T, PublicationTestMode::ACQUIRE_LOAD>(queue);
      publication<T, PublicationTestMode::FAILED_CAS>(queue);
    }

    //TEST - 03
    BOOST_AUTO_TEST_CASE_TEMPLATE(ptx_seq_cst_check, T, IntTypes)
    {
      sycl::queue queue;
      if (!select_cuda(queue) || !supports_type<T>(queue.get_device())) return;

      sequential_consistancy<T, false>(queue);
      sequential_consistancy<T, true>(queue);
    }

    //TEST -04
    BOOST_AUTO_TEST_CASE_TEMPLATE(ptx_cas_contention, T, IntTypes)
    {
      sycl::queue queue;
      if (!select_cuda(queue) || !supports_type<T>(queue.get_device())) return;

      contention<T>(queue);
    }

    //TEST -05
    BOOST_AUTO_TEST_CASE_TEMPLATE(ptx_floating_fetch_min_max, T, FloatTypes)
    {
      sycl::queue queue;
      if (!select_cuda(queue) || !supports_type<T>(queue.get_device())) return;

      fetch_min_max_check<T>(queue);
      publication<T, PublicationTestMode::FLOATING_MIN_MAX>(queue);
    }
    
  }

#endif

BOOST_AUTO_TEST_SUITE_END()
