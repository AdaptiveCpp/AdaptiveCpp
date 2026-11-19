// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=2 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=RESULT
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=4 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=RESULT
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=RESULT
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=RESULT

// Every kind of sub-group collective on the OpenMP device with ACPP_OPENMP_SUBGROUP_SIZE > 1,
// several in one block and in a chain. Work-groups of 12 work-items: at size 8 they give
// sub-groups of 8 and 4 lanes.

#include <algorithm>
#include <climits>
#include <cstdio>
#include <functional>
#include <optional>

#include <sycl/sycl.hpp>

constexpr int W = 12, N = 2 * W;

int x_of(int i) { return i * 37 % 11 - 5; } // -5 .. 5
int y_of(int i) { return i % 3 + 1; }
unsigned u_of(int i) { return i % 2 ? 0x80000000u + i : i; } // signed min/max would differ

// o: F values per work-item; ref(f, l, n, i0): field f of lane l in a sub-group of n lanes whose
// lane 0 is global id i0 (nullopt: unspecified)
using Ref = std::function<std::optional<long long>(int, int, int, int)>;
template <class K> void run(sycl::queue &q, const char *Name, int F, K Kernel, Ref ref) {
  const int S = q.get_device().get_info<sycl::info::device::sub_group_sizes>()[0];
  long long *o = sycl::malloc_shared<long long>(N * F, q);
  q.parallel_for(sycl::nd_range<1>{N, W}, [=](sycl::nd_item<1> it) {
     Kernel(it, o + it.get_global_linear_id() * F);
   }).wait();
  int Bad = 0;
  for (int i = 0; i < N; ++i) {
    const int lid = i % W, l = lid % S, n = std::min(S, W - lid / S * S);
    for (int f = 0; f < F; ++f)
      if (auto e = ref(f, l, n, i - l); e && *e != o[i * F + f] && Bad++ < 5)
        std::printf("%s: field %d, item %d: %lld, expected %lld\n", Name, f, i, o[i * F + f], *e);
  }
  std::printf("%s: %s\n", Name, Bad ? "FAIL" : "ok");
  std::fflush(stdout);
  sycl::free(o, q);
}

// fold op over lanes [0, m) of the sub-group starting at i0
template <class T, class G, class Op> T fold(int i0, int m, G g, T init, Op op) {
  for (int k = 0; k < m; ++k)
    init = op(init, g(i0 + k));
  return init;
}

// all kinds in one block, no collective uses another's result: one step
// RESULT: steps_int: ok
void steps_int(sycl::queue &q) {
  run(q, "steps_int", 19, [](sycl::nd_item<1> it, long long *o) {
    auto sg = it.get_sub_group();
    const int i = it.get_global_linear_id(), x = x_of(i), y = y_of(i), v = x > 3;
    const int l = sg.get_local_linear_id(), n = sg.get_local_linear_range();
    const long long r[] = {
        sycl::shift_group_left(sg, x, 1), sycl::shift_group_right(sg, x, 2),
        sycl::permute_group_by_xor(sg, x, 3), sycl::select_from_group(sg, x, (3 * l + 1) % n),
        sycl::group_broadcast(sg, x, 2), sycl::reduce_over_group(sg, x, sycl::plus<int>{}),
        sycl::reduce_over_group(sg, y, sycl::multiplies<int>{}),
        sycl::reduce_over_group(sg, x, sycl::minimum<int>{}),
        sycl::reduce_over_group(sg, x, sycl::maximum<int>{}),
        sycl::reduce_over_group(sg, x, sycl::bit_and<int>{}),
        sycl::reduce_over_group(sg, x, sycl::bit_or<int>{}),
        sycl::reduce_over_group(sg, x, sycl::bit_xor<int>{}),
        sycl::reduce_over_group(sg, x, sycl::logical_and<int>{}),
        sycl::reduce_over_group(sg, v, sycl::logical_or<int>{}),
        sycl::inclusive_scan_over_group(sg, x, sycl::plus<int>{}),
        sycl::exclusive_scan_over_group(sg, x, sycl::maximum<int>{}),
        sycl::any_of_group(sg, x > 4), sycl::all_of_group(sg, x > -5),
        sycl::none_of_group(sg, x == 5)};
    for (int f = 0; f < 19; ++f)
      o[f] = r[f];
  }, [](int f, int l, int n, int i0) -> std::optional<long long> {
    auto X = [&](int k) { return x_of(i0 + k); };
    auto all = [&](auto g, auto init, auto op) { return fold(i0, n, g, init, op); };
    switch (f) {
    case 0: return l + 1 < n ? std::optional<long long>{X(l + 1)} : std::nullopt;
    case 1: return l >= 2 ? std::optional<long long>{X(l - 2)} : std::nullopt;
    case 2: return (l ^ 3) < n ? std::optional<long long>{X(l ^ 3)} : std::nullopt;
    case 3: return X((3 * l + 1) % n);
    case 4: return 2 < n ? std::optional<long long>{X(2)} : std::nullopt; // lane 2 must exist
    case 5: return all(x_of, 0, std::plus<int>{});
    case 6: return all(y_of, 1, std::multiplies<int>{});
    case 7: return all(x_of, INT_MAX, [](int a, int b) { return std::min(a, b); });
    case 8: return all(x_of, INT_MIN, [](int a, int b) { return std::max(a, b); });
    case 9: return all(x_of, -1, std::bit_and<int>{});
    case 10: return all(x_of, 0, std::bit_or<int>{});
    case 11: return all(x_of, 0, std::bit_xor<int>{});
    case 12: return all(x_of, 1, [](int a, int b) { return a && b; });
    case 13: return all([](int i) { return x_of(i) > 3; }, 0, [](int a, int b) { return a || b; });
    case 14: return fold(i0, l + 1, x_of, 0, std::plus<int>{});
    case 15: return fold(i0, l, x_of, INT_MIN, [](int a, int b) { return std::max(a, b); });
    case 16: return all([](int i) { return x_of(i) > 4; }, false, std::logical_or<bool>{});
    case 17: return all([](int i) { return x_of(i) > -5; }, true, std::logical_and<bool>{});
    default: return !all([](int i) { return x_of(i) == 5; }, false, std::logical_or<bool>{});
    }
  });
}

// unsigned (min / max must compare unsigned) and float (same combine order as the builtins)
// RESULT: steps_uf: ok
void steps_uf(sycl::queue &q) {
  run(q, "steps_uf", 10, [](sycl::nd_item<1> it, long long *o) {
    auto sg = it.get_sub_group();
    const int i = it.get_global_linear_id();
    const unsigned u = u_of(i);
    const float x = x_of(i), y = y_of(i);
    const unsigned ru[] = {sycl::reduce_over_group(sg, u, sycl::minimum<unsigned>{}),
                           sycl::reduce_over_group(sg, u, sycl::maximum<unsigned>{}),
                           sycl::inclusive_scan_over_group(sg, u, sycl::maximum<unsigned>{})};
    const float rf[] = {sycl::reduce_over_group(sg, x, sycl::plus<float>{}),
                        sycl::reduce_over_group(sg, x, sycl::minimum<float>{}),
                        sycl::reduce_over_group(sg, x, sycl::maximum<float>{}),
                        sycl::reduce_over_group(sg, y, sycl::multiplies<float>{}),
                        sycl::inclusive_scan_over_group(sg, x, sycl::plus<float>{}),
                        sycl::exclusive_scan_over_group(sg, x, sycl::plus<float>{}),
                        sycl::shift_group_right(sg, x, 1)};
    for (int f = 0; f < 3; ++f)
      o[f] = ru[f];
    for (int f = 0; f < 7; ++f)
      o[3 + f] = rf[f];
  }, [](int f, int l, int n, int i0) -> std::optional<long long> {
    auto umin = [](unsigned a, unsigned b) { return std::min(a, b); };
    auto umax = [](unsigned a, unsigned b) { return std::max(a, b); };
    switch (f) {
    case 0: return fold(i0, n, u_of, UINT_MAX, umin);
    case 1: return fold(i0, n, u_of, 0u, umax);
    case 2: return fold(i0, l + 1, u_of, 0u, umax);
    case 3: return fold(i0, n, x_of, 0, std::plus<int>{});
    case 4: return fold(i0, n, x_of, INT_MAX, [](int a, int b) { return std::min(a, b); });
    case 5: return fold(i0, n, x_of, INT_MIN, [](int a, int b) { return std::max(a, b); });
    case 6: return fold(i0, n, y_of, 1, std::multiplies<int>{});
    case 7: return fold(i0, l + 1, x_of, 0, std::plus<int>{});
    case 8: return fold(i0, l, x_of, 0, std::plus<int>{});
    default: return l >= 1 ? std::optional<long long>{x_of(i0 + l - 1)} : std::nullopt;
    }
  });
}

// b uses a's result but is pure: it moves after c, so a and c share a step; the reduce needs b:
// a second step
// RESULT: steps_chain: ok
void steps_chain(sycl::queue &q) {
  run(q, "steps_chain", 3, [](sycl::nd_item<1> it, long long *o) {
    auto sg = it.get_sub_group();
    const int x = x_of(it.get_global_linear_id());
    const int a = sycl::permute_group_by_xor(sg, x, 1);
    const int b = a + x;
    const int c = sycl::shift_group_right(sg, x, 1);
    const int d = sycl::reduce_over_group(sg, b, sycl::plus<int>{});
    o[0] = a;
    o[1] = c;
    o[2] = d;
  }, [](int f, int l, int n, int i0) -> std::optional<long long> {
    switch (f) {
    case 0: return x_of(i0 + (l ^ 1)); // n is even
    case 1: return l >= 1 ? std::optional<long long>{x_of(i0 + l - 1)} : std::nullopt;
    default: return 2 * fold(i0, n, x_of, 0, std::plus<int>{});
    }
  });
}

int main() {
  sycl::queue q;
  steps_int(q);
  steps_uf(q);
  steps_chain(q);
}
