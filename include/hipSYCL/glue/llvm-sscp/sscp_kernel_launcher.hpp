/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP

#include "hipSYCL/glue/generic/code_object.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "ir_constants.hpp"

// TODO: Maybe this can be unified with the HIPSYCL_STATIC_HCF_REGISTRATION
// macro. We cannot use this macro directly because it expects
// the object id to be constexpr, which it is not for the SSCP case.
struct __hipsycl_static_sscp_hcf_registration {
  __hipsycl_static_sscp_hcf_registration() {
    ::hipsycl::rt::kernel_cache::get().register_hcf_object(
        ::hipsycl::common::hcf_container{std::string{
            reinterpret_cast<const char *>(__hipsycl_local_sscp_hcf_content),
            __hipsycl_local_sscp_hcf_object_size}});
  }
};
static __hipsycl_static_sscp_hcf_registration
    __hipsycl_register_sscp_hcf_object;

#define __sycl_kernel [[clang::annotate("hipsycl_sscp_kernel")]]

namespace hipsycl {
namespace glue {
namespace sscp_dispatch {

template <typename KernelName, typename KernelType>
__sycl_kernel void
kernel_single_task(const KernelType &kernel) {
  kernel();
}

template <typename KernelName, typename KernelType>
__sycl_kernel void
kernel_parallel_for(const KernelType& kernel) {
  kernel();
}

}


class sscp_kernel_launcher : public rt::backend_kernel_launcher
{
public:

  sscp_kernel_launcher() {}
  virtual ~sscp_kernel_launcher(){}

  virtual void set_params(void*) override {}

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {

    this->_type = type;
    this->_invoker = [=] (rt::dag_node* node) mutable {

      static_cast<rt::kernel_operation *>(node->get_operation())
          ->initialize_embedded_pointers(k, reductions...);

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      auto get_grid_range = [&]() {
        for (int i = 0; i < Dim; ++i){
          if (global_range[i] % local_range[i] != 0) {
            rt::register_error(__hipsycl_here(),
                               rt::error_info{"sscp_dispatch: global range is "
                                              "not divisible by local range"});
          }
        }

        return global_range / local_range;
      };

      if constexpr(type == rt::kernel_type::single_task){

        sscp_dispatch::kernel_single_task(k);

      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

        

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {


      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

        
      } else if constexpr (type == rt::kernel_type::custom) {
        // TODO
      }
      else {
        assert(false && "Unsupported kernel type");
      }

    };
  }

  virtual rt::backend_id get_backend() const final override {
    // TODO
    return rt::backend_id::omp;
  }

  virtual void invoke(rt::dag_node* node) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:

  std::function<void (rt::dag_node*)> _invoker;
  rt::kernel_type _type;
};


}
}

#undef __sycl_kernel

#endif
