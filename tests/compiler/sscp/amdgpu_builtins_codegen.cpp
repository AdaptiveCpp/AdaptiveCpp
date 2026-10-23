// REQUIRES: amdgpu-backend-tools
// RUN: %acpp %s -c -o %t.o -mllvm -acpp-sscp-emit-hcf
// RUN: %llvm-to-amdgpu --ir %s.hcf %t.bc llvm-ir.global
// RUN: rm -f %s.hcf
// RUN: %llvm-dis %t.bc -o %t.ll
// RUN: FileCheck --check-prefix=CHECK-IR %s < %t.ll
// RUN: %clangxx -target amdgcn-amd-amdhsa -mcpu=gfx90a -S %t.ll -o %t.s
// RUN: FileCheck --check-prefix=CHECK-ASM %s < %t.s

#include <sycl/sycl.hpp>
#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu_builtins.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"

namespace jit = hipsycl::sycl::AdaptiveCpp_jit;

// CHECK-IR: define amdgpu_kernel void @{{.*}}basic_parallel_for{{.*}}

void test_builtins(float *dev_f32, double *dev_f64,
                   int *dev_dpp_shr, int *dev_dpp_ror, int *dev_dpp_mirror,
                   int *dev_rfl, float *dev_fract, int global_idx) {
  __acpp_if_target_sscp(
    jit::compile_if(
      jit::reflect<jit::reflection_query::target_arch>() ==
        adaptivecpp::amdgpu::kGfx90aArchId,
      [&]() {
        // ── Floating-point atomic add ───────────────────────────────────────
        // CHECK-IR: atomicrmw fadd ptr addrspace(1) {{.*}} !amdgpu.no.fine.grained.memory ![[MD:[0-9]+]], !amdgpu.ignore.denormal.mode ![[MD]]
        // CHECK-ASM: global_atomic_add_f32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}]
        adaptivecpp::amdgpu::unsafe_atomic_fetch_add(
          hipsycl::sycl::access::address_space::global_space,
          sycl::memory_order::relaxed,
          hipsycl::sycl::memory_scope::device,
          dev_f32, 1.5f);

        // CHECK-IR: atomicrmw fadd ptr addrspace(1) {{.*}} !amdgpu.no.fine.grained.memory ![[MD]], !amdgpu.ignore.denormal.mode ![[MD]]
        // CHECK-ASM: global_atomic_add_f64 v{{.*}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}]
        adaptivecpp::amdgpu::unsafe_atomic_fetch_add(
          hipsycl::sycl::access::address_space::global_space,
          sycl::memory_order::relaxed,
          hipsycl::sycl::memory_scope::device,
          dev_f64, 2.5);

        // ── DPP: row_shr:1 (0x111), bound_ctrl=true ────────────────────────
        // CHECK-IR: call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 {{.*}}, i32 273, i32 15, i32 15, i1 true)
        // CHECK-ASM: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        *dev_dpp_shr = adaptivecpp::amdgpu::update_dpp<0x111, 0xf, 0xf, true>(global_idx);

        // ── DPP: row_ror:1 (0x121), bound_ctrl=false ───────────────────────
        // CHECK-IR: call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 {{.*}}, i32 289, i32 15, i32 15, i1 false)
        // CHECK-ASM: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_ror:1 row_mask:0xf bank_mask:0xf
        *dev_dpp_ror = adaptivecpp::amdgpu::update_dpp<0x121, 0xf, 0xf, false>(global_idx);

        // ── DPP: row_mirror (0x140), bound_ctrl=false ──────────────────────
        // CHECK-IR: call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 {{.*}}, i32 320, i32 15, i32 15, i1 false)
        // CHECK-ASM: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_mirror row_mask:0xf bank_mask:0xf
        *dev_dpp_mirror = adaptivecpp::amdgpu::update_dpp<0x140, 0xf, 0xf, false>(global_idx);

        // ── readfirstlane ───────────────────────────────────────────────────
        // CHECK-IR: call i32 @llvm.amdgcn.readfirstlane.i32(i32
        // CHECK-ASM: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
        *dev_rfl = adaptivecpp::amdgpu::readfirstlane(global_idx);

        // ── fract ───────────────────────────────────────────────────────────
        // CHECK-IR: call float @llvm.amdgcn.fract.f32(float
        // CHECK-ASM: v_fract_f32_e64 v{{[0-9]+}}, v{{[0-9]+}}
        *dev_fract = adaptivecpp::amdgpu::fract(static_cast<float>(global_idx) + 0.456f);
      }
    );
  );
}

int main() {
  sycl::queue q;

  float  *dev_f32        = sycl::malloc_device<float> (1024, q);
  double *dev_f64        = sycl::malloc_device<double>(1024, q);
  int    *dev_dpp_shr    = sycl::malloc_device<int>   (1024, q);
  int    *dev_dpp_ror    = sycl::malloc_device<int>   (1024, q);
  int    *dev_dpp_mirror = sycl::malloc_device<int>   (1024, q);
  int    *dev_rfl        = sycl::malloc_device<int>   (1024, q);
  float  *dev_fract      = sycl::malloc_device<float> (1024, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> idx) {
      test_builtins(dev_f32, dev_f64,
                    dev_dpp_shr, dev_dpp_ror, dev_dpp_mirror,
                    dev_rfl, dev_fract, idx[0]);
    });
  }).wait();

  sycl::free(dev_f32,        q);
  sycl::free(dev_f64,        q);
  sycl::free(dev_dpp_shr,    q);
  sycl::free(dev_dpp_ror,    q);
  sycl::free(dev_dpp_mirror, q);
  sycl::free(dev_rfl,        q);
  sycl::free(dev_fract,      q);
}
