/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/ocl/ocl_hardware_manager.hpp"
#include "hipSYCL/runtime/ocl/ocl_usm.hpp"
#include "hipSYCL/runtime/operations.hpp"

#include <CL/opencl.hpp>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {
template<int Query, class ResultT>
ResultT info_query(const cl::Device& dev) {
  ResultT r{};
  cl_int err = dev.getInfo(Query, &r);
  if(err != CL_SUCCESS) {
    register_error(
          __hipsycl_here(),
          error_info{"ocl_usm: Could not obtain device info",
                    error_code{"CL", err}});
  }
  return r;
}

}

class ocl_usm_intel_extension : public ocl_usm {
public:
  ocl_usm_intel_extension(ocl_hardware_manager *hw_mgr, int device_index,
                          const cl::Platform &platform, const cl::Device &dev,
                          const cl::Context &ctx)
      : _ctx{ctx}, _dev{dev}, _hw_mgr{hw_mgr} {
    std::string str;
    cl_int err = dev.getInfo(CL_DEVICE_EXTENSIONS, &str);
    if (err == CL_SUCCESS &&
        (str.find("cl_intel_unified_shared_memory") != std::string::npos)) {
      _is_available = true;

      cl_platform_id id = platform.cl::detail::Wrapper<cl_platform_id>::get();

      initialize_func(_host_mem_alloc, "clHostMemAllocINTEL", id);
      initialize_func(_device_mem_alloc, "clDeviceMemAllocINTEL", id);
      initialize_func(_shared_mem_alloc, "clSharedMemAllocINTEL", id);
      initialize_func(_free, "clMemFreeINTEL", id);
      initialize_func(_blocking_free, "clMemBlockingFreeINTEL", id);
      initialize_func(_mem_alloc_info, "clGetMemAllocInfoINTEL", id);
      initialize_func(_set_kernel_arg_mem_pointer,
                      "clSetKernelArgMemPointerINTEL", id);
      initialize_func(_mem_fill, "clEnqueueMemFillINTEL", id);
      initialize_func(_mem_copy, "clEnqueueMemcpyINTEL", id);
      initialize_func(_mem_advise, "clEnqueueMemAdviseINTEL", id);
      initialize_func(_migrate_mem, "clEnqueueMigrateMemINTEL", id);

      // On CPU, we can be more relaxed in a couple of places as
      // USM will generally "just work"
      _is_cpu = _hw_mgr->get_device(device_index)->is_cpu();
    }
  }

  virtual bool is_available() const override {
    return _is_available;
  }

  virtual bool has_usm_device_allocations() const override {
    if(!_is_available)
      return false;
    return info_query<CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, cl_bitfield>(_dev) != CL_NONE;
  }

  virtual bool has_usm_host_allocations() const override {
    if(!_is_available)
      return false;
    return info_query<CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL, cl_bitfield>(_dev) != CL_NONE;
  }

  virtual bool has_usm_atomic_host_allocations() const override {
    if(!_is_available)
      return false;
    return (info_query<CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, cl_bitfield>(
               _dev) &
           CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL) != 0;
  }

  virtual bool has_usm_shared_allocations() const override {
    if(!_is_available)
      return false;
    return info_query<CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, cl_bitfield>(_dev) != CL_NONE;
  }

  virtual bool has_usm_atomic_shared_allocations() const override {
    if (!_is_available)
      return false;
    return info_query<CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
                      cl_bitfield>(_dev) &
           CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL;
  }

  virtual bool has_usm_system_allocations() const override {
    if(!_is_available)
      return false;
    return info_query<CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL, cl_bitfield>(_dev) != CL_NONE;
  }

  virtual void* malloc_host(std::size_t size, std::size_t alignment, cl_int& err) override {
    if(!_host_mem_alloc) {
      err = CL_INVALID_PLATFORM;
      return nullptr;
    }
    cl_mem_properties_intel props = 0;
    return _host_mem_alloc(_ctx.get(), &props, size, alignment, &err);
  }

  virtual void* malloc_device(std::size_t size, std::size_t alignment, cl_int& err) override {
    if(!_device_mem_alloc) {
      err = CL_INVALID_PLATFORM;
      return nullptr;
    }
    cl_mem_properties_intel props = 0;
    return _device_mem_alloc(_ctx.get(), _dev.get(), &props, size, alignment,
                             &err);
  }

  virtual void* malloc_shared(std::size_t size, std::size_t alignment, cl_int& err) override {
    if(!_shared_mem_alloc) {
      err = CL_INVALID_PLATFORM;
      return nullptr;
    }
    cl_mem_properties_intel props = 0;
    return _shared_mem_alloc(_ctx.get(), _dev.get(), &props, size, alignment,
                             &err);
  }

  virtual cl_int free(void* ptr) override {
    if(!_free) {
      return CL_INVALID_PLATFORM;
    }
    return _free(_ctx.get(), ptr);
  }

  virtual cl_int get_alloc_info(const void* ptr, pointer_info& out) override {
    if(!_mem_alloc_info) {
      return CL_INVALID_PLATFORM;
    }

    cl_unified_shared_memory_type_intel mem_type;
    cl_int err = _mem_alloc_info(_ctx.get(), ptr, CL_MEM_ALLOC_TYPE_INTEL,
                                 sizeof(mem_type), &mem_type, nullptr);

    out.is_from_host_backend = false;
    out.is_optimized_host = false;

    if(err != CL_SUCCESS)
      return err;

    if(mem_type == CL_MEM_TYPE_HOST_INTEL)
      out.is_optimized_host = true;
    else if(mem_type == CL_MEM_TYPE_SHARED_INTEL)
      out.is_usm = true;
    else if(mem_type == CL_MEM_TYPE_DEVICE_INTEL) {
      cl_device_id dev;
      err = _mem_alloc_info(_ctx.get(), ptr, CL_MEM_ALLOC_DEVICE_INTEL,
                                 sizeof(dev), &dev, nullptr);

      if(err != CL_SUCCESS)
        return err;
      
      bool found = false;
      for(std::size_t i = 0; i < _hw_mgr->get_num_devices(); ++i) {
        ocl_hardware_context* ctx = static_cast<ocl_hardware_context*>(_hw_mgr->get_device(i));
        if(ctx->get_cl_device().get() == dev) {
          found = true;
          out.dev = _hw_mgr->get_device_id(i);
        }
      }
      if(!found) {
        return CL_INVALID_MEM_OBJECT;
      }
    } else {
      return CL_INVALID_MEM_OBJECT;
    }

    return CL_SUCCESS;
  }

  virtual cl_int enqueue_memcpy(cl::CommandQueue &queue, void *dst,
                                const void *src, std::size_t size,
                                const std::vector<cl::Event> &wait_events,
                                cl::Event *event) override {
    if(!_mem_copy)
      return CL_INVALID_PLATFORM;
    
    cl_event tmp;
    cl_int err = _mem_copy(
        queue.get(), false, dst, src, size, wait_events.size(),
        (wait_events.size() > 0) ? (cl_event *)&wait_events.front() : nullptr,
        (event != nullptr) ? &tmp : nullptr);
    if(event != nullptr && err == CL_SUCCESS)
      *event = tmp;
    
    return err;
  }



  cl_int enqueue_memset(cl::CommandQueue &queue, void *ptr,
                                cl_int pattern, std::size_t bytes,
                                const std::vector<cl::Event> &wait_events,
                                cl::Event *event) override {
    
    unsigned char pattern_byte = static_cast<char>(pattern);

    cl_event tmp;
    cl_int err = _mem_fill(
        queue.get(), ptr, &pattern_byte, 1 /*pattern size*/, bytes,
        wait_events.size(),
        (wait_events.size() > 0) ? (cl_event *)&wait_events.front() : nullptr,
        (event != nullptr) ? &tmp : nullptr);
    if(event != nullptr && err == CL_SUCCESS) {
      *event = tmp;
    }
    return err;
  }

  cl_int enqueue_prefetch(cl::CommandQueue &queue, const void *ptr,
                          std::size_t bytes,
                          cl_mem_migration_flags flags,
                          const std::vector<cl::Event> &wait_events,
                          cl::Event *event) override {
    cl_event tmp;
    cl_int err = _migrate_mem(
        queue.get(), ptr, bytes, flags, wait_events.size(),
        (wait_events.size() > 0) ? (cl_event *)&wait_events.front() : nullptr,
        (event != nullptr) ? &tmp : nullptr);

    if(event != nullptr && err == CL_SUCCESS) {
      *event = tmp;
    }
    return err;
  }

  cl_int enable_indirect_usm_access(cl::Kernel& k) override {
    auto maybe_ignore = [this](cl_int error_code) {
      // Intel CPU OpenCL seems to not understand these flags. We can just ignore USM errors
      // on CPUs anyway.
      if(_is_cpu)
        return CL_SUCCESS;
      return error_code;
    };

    cl_int err = maybe_ignore(
        k.setExecInfo(CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, cl_bool{true}));

    if(err != CL_SUCCESS)
      return err;
    err = maybe_ignore(k.setExecInfo(CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, cl_bool{true}));
    if(err != CL_SUCCESS)
      return err;
    err = maybe_ignore(k.setExecInfo(CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, cl_bool{true}));

    return err;
  }

private:
  template <class Func>
  void initialize_func(Func &out, const char *name, cl_platform_id id) {
    out = (Func)clGetExtensionFunctionAddressForPlatform(id, name);
    if (!out) {
      print_error(
          __hipsycl_here(),
          error_info{"ocl_usm_extension: Platform advertises USM support, but "
                     "extracting function address for " +
                     std::string{name} + " failed."});
    }
  }

  bool _is_available = false;
  clHostMemAllocINTEL_fn _host_mem_alloc = nullptr;
  clDeviceMemAllocINTEL_fn _device_mem_alloc = nullptr;
  clSharedMemAllocINTEL_fn _shared_mem_alloc = nullptr;
  clMemFreeINTEL_fn _free = nullptr;
  clMemBlockingFreeINTEL_fn _blocking_free = nullptr;
  clGetMemAllocInfoINTEL_fn _mem_alloc_info = nullptr;
  clSetKernelArgMemPointerINTEL_fn _set_kernel_arg_mem_pointer = nullptr;
  clEnqueueMemFillINTEL_fn _mem_fill = nullptr;
  clEnqueueMemcpyINTEL_fn _mem_copy = nullptr;
  clEnqueueMemAdviseINTEL_fn _mem_advise = nullptr;
  clEnqueueMigrateMemINTEL_fn _migrate_mem = nullptr;

  cl::Context _ctx;
  cl::Device _dev;
  ocl_hardware_manager* _hw_mgr;
  bool _is_cpu = false;
};



class ocl_usm_svm : public ocl_usm {
public:
  ocl_usm_svm(ocl_hardware_manager *hw_mgr, int device_index,
              const cl::Platform &platform, const cl::Device &dev,
              const cl::Context &ctx)
      : _ctx{ctx}, _dev{dev}, _hw_mgr{hw_mgr}, _device_index{device_index} {

    _is_cpu = _hw_mgr->get_device(device_index)->is_cpu();

    cl_device_svm_capabilities cap;
    cl_int err = dev.getInfo(CL_DEVICE_SVM_CAPABILITIES, &cap);
    // Need at least fine-grained system for proper USM semantics,
    // as SVM otherwise requires us to specify all pointers a kernel uses.
    // This we cannot know in general, however.
    
    if (err == CL_SUCCESS && (cap & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
      _is_available = true;
      if(cap & (cap & CL_DEVICE_SVM_ATOMICS))
        _has_atomics = true;
    }
  }

  virtual bool is_available() const override {
    return _is_available;
  }

  virtual bool has_usm_device_allocations() const override {
    return has_usm_shared_allocations();
  }

  virtual bool has_usm_host_allocations() const override {
    return has_usm_shared_allocations();
  }

  virtual bool has_usm_atomic_host_allocations() const override {
    return has_usm_atomic_shared_allocations();
  }

  virtual bool has_usm_shared_allocations() const override {
    return _is_available;
  }

  virtual bool has_usm_atomic_shared_allocations() const override {
    if (!_is_available)
      return false;
    
    return _has_atomics;
  }

  virtual bool has_usm_system_allocations() const override {
    return _is_available;
  }

  virtual void* malloc_host(std::size_t size, std::size_t alignment, cl_int& err) override {
    return this->malloc_shared(size, alignment, err);
  }

  virtual void* malloc_device(std::size_t size, std::size_t alignment, cl_int& err) override {
    return this->malloc_shared(size, alignment, err);
  }

  virtual void* malloc_shared(std::size_t size, std::size_t alignment, cl_int& err) override {
    if(!_is_available) {
      err = CL_INVALID_PLATFORM;
      return nullptr;
    }
    
    void* ptr = clSVMAlloc(_ctx.get(),
                      CL_MEM_SVM_FINE_GRAIN_BUFFER & CL_DEVICE_SVM_ATOMICS,
                      size, static_cast<cl_uint>(alignment));
    if(ptr)
      err = CL_SUCCESS;
    else
      err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return ptr;
  }

  virtual cl_int free(void* ptr) override {
    if(!_is_available) {
      return CL_INVALID_PLATFORM;
    }
    clSVMFree(_ctx.get(), ptr);
    return CL_SUCCESS;
  }


  virtual cl_int get_alloc_info(const void* ptr, pointer_info& out) override {
    if(!_is_available) {
      return CL_INVALID_PLATFORM;
    }

    // TODO: We don't have a direct way of querying where a SVM pointer
    // comes from, so for now we just assume everything is a SVM pointer from
    // this device with shared USM semantics. This might not be correct
    // if other devices are used too.
    out.is_from_host_backend = false;
    out.is_usm = true;
    out.is_optimized_host = false;
    out.dev = _hw_mgr->get_device_id(_device_index);

    return CL_SUCCESS;
  }

  virtual cl_int enqueue_memcpy(cl::CommandQueue &queue, void *dst,
                                const void *src, std::size_t size,
                                const std::vector<cl::Event> &wait_events,
                                cl::Event *evt_out) override {
    return queue.enqueueMemcpySVM(dst, src, false, size, &wait_events, evt_out);
  }



  cl_int enqueue_memset(cl::CommandQueue &queue, void *ptr,
                                cl_int pattern, std::size_t bytes,
                                const std::vector<cl::Event> &wait_events,
                                cl::Event *out) override {
    unsigned char pattern_byte = static_cast<char>(pattern);
    return queue.enqueueMemFillSVM(ptr, pattern_byte, bytes, &wait_events, out);
  }


  cl_int enqueue_prefetch(cl::CommandQueue &queue, const void *ptr,
                          std::size_t bytes,
                          cl_mem_migration_flags flags,
                          const std::vector<cl::Event> &wait_events,
                          cl::Event *event) override {
    // Seems there is a bug in CommandQueue::enqueueMigrateSVM, so we directly
    // call the OpenCL function
    cl_event tmp;
    cl_int err = ::clEnqueueSVMMigrateMem(
        queue.get(), 1, &ptr, &bytes, flags, wait_events.size(),
        (wait_events.size() > 0) ? (cl_event *)&wait_events.front() : nullptr,
        (event != nullptr) ? &tmp : nullptr);
    
    if(event != nullptr && err == CL_SUCCESS) {
      *event = tmp;
    }
    return err;
  }


  cl_int enable_indirect_usm_access(cl::Kernel& k) override {
    return k.setExecInfo(CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM, cl_bool{true});
  }

private:
  bool _is_available = false;
  cl::Context _ctx;
  cl::Device _dev;
  ocl_hardware_manager* _hw_mgr;
  bool _is_cpu = false;
  bool _has_atomics = false;
  int _device_index;
};

std::unique_ptr<ocl_usm>
ocl_usm::from_intel_extension(ocl_hardware_manager* hw_mgr, int dev_id) {
  
  ocl_hardware_context *ctx =
      static_cast<ocl_hardware_context *>(hw_mgr->get_device(dev_id));
  int platform_id = ctx->get_platform_id();
  return std::make_unique<ocl_usm_intel_extension>(
      hw_mgr, dev_id, hw_mgr->get_platform(platform_id), ctx->get_cl_device(),
      ctx->get_cl_context());
}

std::unique_ptr<ocl_usm>
ocl_usm::from_fine_grained_system_svm(ocl_hardware_manager* hw_mgr, int dev_id) {
  
  ocl_hardware_context *ctx =
      static_cast<ocl_hardware_context *>(hw_mgr->get_device(dev_id));
  int platform_id = ctx->get_platform_id();
  return std::make_unique<ocl_usm_svm>(
      hw_mgr, dev_id, hw_mgr->get_platform(platform_id), ctx->get_cl_device(),
      ctx->get_cl_context());
}

}
}
