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
#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
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


template <typename KernelType>
// hipsycl_sscp_kernel causes kernel entries to be emitted to the HCF
[[clang::annotate("hipsycl_sscp_kernel")]]
// hipsycl_sscp_outlining creates an entrypoint for outlining of device code
[[clang::annotate("hipsycl_sscp_outlining")]]
void __hipsycl_sscp_kernel(const KernelType kernel) {
  if(__hipsycl_sscp_is_device)
    kernel();
}

// The SSCP compiler will use this invocation to connect the kernel function
// to a global variable where the kernel name can be stored.
// First argument has to be a function pointer to the kernel,
// second one a pointer to a global variable.
// No indirection is allowed! If I say, the argument has to be a global variable,
// I mean it. Directly. No passing through other functions first.
template <class Kernel>
void __hipsycl_sscp_extract_kernel_name(void (*Func)(Kernel),
                                        const char *target);

template<class Kernel>
struct __hipsycl_sscp_kernel_name {
  static const char value [];
};

namespace hipsycl {
namespace glue {
namespace sscp_dispatch {

template <int Dimensions, bool WithOffset>
bool item_is_in_range(const sycl::item<Dimensions, WithOffset> &item,
                      const sycl::range<Dimensions> &execution_range,
                      const sycl::id<Dimensions>& offset = {}) {

  for(int i = 0; i < Dimensions; ++i) {
    if constexpr(WithOffset) {
      if(item.get_id(i) >= offset.get(i) + execution_range.get(i))
        return false;
    } else {
      if(item.get_id(i) >= execution_range.get(i))
        return false;
    }
  }
  return true;
}

template<class UserKernel>
class single_task {
public:
  single_task(const UserKernel& k)
  : _k{k} {}

  void operator()() const {
    _k();
  }
private:
  UserKernel _k;
};

template<class UserKernel, int Dimensions>
class basic_parallel_for {
public:
  basic_parallel_for(const UserKernel &k,
                     sycl::range<Dimensions> execution_range)
      : _k{k}, _range{execution_range} {}

  void operator()() const {
    auto this_item = sycl::detail::make_item<Dimensions>(
      sycl::detail::get_global_id<Dimensions>(), _range
    );
    if(item_is_in_range(this_item, _range))
      _k(this_item);
  }
private:
  UserKernel _k;
  sycl::range<Dimensions> _range;
};

template<class UserKernel, int Dimensions>
class basic_parallel_for_offset {
public:
  basic_parallel_for_offset(const UserKernel &k, sycl::id<Dimensions> offset,
                            sycl::range<Dimensions> execution_range)
      : _k{k}, _range{execution_range}, _offset{offset} {}

  void operator()() const {
    auto this_item = sycl::detail::make_item<Dimensions>(
        sycl::detail::get_global_id<Dimensions>() + _offset, _range, _offset);
    
    if(item_is_in_range(this_item, _range, _offset))
      _k(this_item);
  }
private:
  UserKernel _k;
  sycl::range<Dimensions> _range;
  sycl::id<Dimensions> _offset;
};


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

      std::string kernel_name;

      if constexpr(type == rt::kernel_type::single_task){
        generate_kernel(sscp_dispatch::single_task{k}, kernel_name);
      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {
        if(offset == sycl::id<Dim>{}) {
          generate_kernel(sscp_dispatch::basic_parallel_for{k, global_range}, kernel_name);
        } else {
          generate_kernel(sscp_dispatch::basic_parallel_for_offset{k, offset, global_range}, kernel_name);
        }
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

  virtual int get_backend_score(rt::backend_id b) const final override {
    // The other backends return 2 for exact matches,
    // so this means that SSCP is currently preferred when no
    // other exactly matching backend kernel launcher was found.
    // TODO: Should we prevent selection of SSCP if the backend
    // does not support SSCP runtime compilation?
    return 1;
  }

  virtual void invoke(rt::dag_node* node) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:
  template<class Kernel, int Dim>
  void launch_kernel() {

  }

  template<class Kernel>
  static void generate_kernel(const Kernel& k, std::string& kernel_name) {
    if (__hipsycl_sscp_is_device) {
      __hipsycl_sscp_kernel(k);
    }
    
    __hipsycl_sscp_extract_kernel_name<Kernel>(
        &__hipsycl_sscp_kernel<Kernel>,
        &__hipsycl_sscp_kernel_name<Kernel>::value[0]);
    kernel_name = std::string{&__hipsycl_sscp_kernel_name<Kernel>::value[0]};
  }

  std::function<void (rt::dag_node*)> _invoker;
  rt::kernel_type _type;
};


}
}

#endif
