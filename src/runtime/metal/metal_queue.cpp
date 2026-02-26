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
#include "hipSYCL/runtime/metal/metal_queue.hpp"
#include "hipSYCL/runtime/metal/metal_code_object.hpp"
#include "hipSYCL/runtime/metal/metal_event.hpp"
#include "hipSYCL/compiler/llvm-to-backend/metal/LLVMToMetalFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/common/debug.hpp"

#include <Metal/Metal.hpp>

#undef nil

namespace hipsycl {
namespace rt {

namespace {

// Metal supports at most 31 [[buffer(N)]] arguments in flat mode.
// When a kernel has more than metal_max_args_for_flat_mode parameters,
// all arguments are packed into a single argument buffer struct instead.
// This value is passed to the compiler via kernel_build_option::metal_max_args_for_flat_mode,
// which sets MetalEmitterOptions::maxArgsForFlatMode in LLVMToMetal.
static constexpr int metal_max_args_for_flat_mode = 6;

inline unsigned align_up(unsigned x, unsigned a) {
  return (x + (a - 1)) & ~(a - 1);
}

void encode_arguments(
  MTL::ComputeCommandEncoder* encoder,
  MTL::Device* device,
  metal_allocator* allocator,
  void** args,
  std::size_t* arg_sizes,
  std::size_t num_args,
  const std::vector<int>& is_pointer_arg
) {
  for (std::size_t i = 0; i < num_args; ++i) {
    if (is_pointer_arg[i]) {
      void* usm_ptr = *(void**)args[i];
      auto [buffer, offset, _] = allocator->get_usm_block(usm_ptr);
      if (buffer) {
        encoder->setBuffer(buffer, offset, i);
      }
    } else {
      encoder->setBytes(args[i], arg_sizes[i], i);
    }
  }
}

void encode_arguments_argbuffer(
  MTL::ComputeCommandEncoder* encoder,
  MTL::Device* device,
  metal_allocator* allocator,
  MTL::Function* function,
  void** args,
  std::size_t* arg_sizes,
  std::size_t num_args,
  const std::vector<int>& is_pointer_arg,
  std::vector<NS::SharedPtr<MTL::Buffer>>& buffers_out
) {
  NS::SharedPtr<MTL::ArgumentEncoder> arg_enc = NS::TransferPtr(function->newArgumentEncoder(0));

  const size_t arg_len = arg_enc->encodedLength();
  auto arg_buffer_out = buffers_out.emplace_back(NS::TransferPtr(device->newBuffer(arg_len, MTL::ResourceStorageModeShared)));

  arg_enc->setArgumentBuffer(arg_buffer_out.get(), 0);

  for (std::size_t i = 0; i < num_args; ++i) {
    if (is_pointer_arg[i]) {
      void* usm_ptr = *(void**)args[i];
      auto [buffer, offset, _] = allocator->get_usm_block(usm_ptr);
      if (buffer) {
        arg_enc->setBuffer(buffer, offset, i);
        encoder->useResource(buffer, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      }
    } else {
      auto* dst = arg_enc->constantData(i);
      std::memcpy(dst, args[i], arg_sizes[i]);
    }
  }

  encoder->setBuffer(arg_buffer_out.get(), 0, 0);
}

result launch_kernel_from_library(
  MTL::Library* library,
  MTL::Device* device,
  metal_allocator* allocator,
  std::string_view kernel_name,
  const rt::range<3>& num_groups,
  const rt::range<3>& group_size,
  unsigned local_mem_size,
  void** args,
  std::size_t* arg_sizes,
  std::size_t num_args,
  const rt::hcf_kernel_info* kernel_info,
  const std::optional<std::vector<int>>& retained_indices)
{
  if (!library) {
    return make_error(__acpp_here(),
                      error_info{"metal: Library is null"});
  }

  auto entry = std::string(kernel_name);
  // Get the kernel function from library
  NS::String* function_name = NS::String::string(entry.c_str(), NS::UTF8StringEncoding);
  NS::SharedPtr<MTL::Function> function = NS::TransferPtr(library->newFunction(function_name));

  if (!function) {
    return make_error(__acpp_here(),
                      error_info{"metal: Could not find kernel function: " +
                                 std::string(kernel_name)});
  }

  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_state = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));

  if (error || !pipeline_state) {
    std::string error_msg = "metal: Failed to create compute pipeline state";
    if (error && error->localizedDescription()) {
      error_msg += ": ";
      error_msg += error->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{error_msg});
  }

  std::vector<int> is_pointer_arg(num_args, 0);
  for (std::size_t i = 0; i < num_args; ++i) {
    std::size_t ki = retained_indices.has_value() ? retained_indices.value()[i] : i;
    if (ki < kernel_info->get_num_parameters()) {
      is_pointer_arg[i] = (kernel_info->get_argument_type(ki) == rt::hcf_kernel_info::pointer) ? 1 : 0;
    }
  }
  bool arg_buffer_used = static_cast<int>(num_args) > metal_max_args_for_flat_mode;

  NS::SharedPtr<MTL::CommandQueue> command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    return make_error(__acpp_here(),
                      error_info{"metal: Failed to create command queue"});
  }

  auto* command_buffer = command_queue->commandBuffer();
  if (!command_buffer) {
    return make_error(__acpp_here(),
                      error_info{"metal: Failed to create command buffer"});
  }

  auto* encoder = command_buffer->computeCommandEncoder();
  if (!encoder) {
    return make_error(__acpp_here(),
                      error_info{"metal: Failed to create compute encoder"});
  }

  encoder->setComputePipelineState(pipeline_state.get());
  auto user_local_mem_size = local_mem_size;
  auto threads_in_group = num_groups[0] * num_groups[1] * num_groups[2];
  // TODO: check if workgroup reduction is used in kernel_info to adjust local mem size
  auto additional_local_mem_per_thread = (threads_in_group + 32 - 1) / 32; // for per-workgroup reductions

  local_mem_size = user_local_mem_size + additional_local_mem_per_thread;
  if (local_mem_size != 0) {
    encoder->setThreadgroupMemoryLength(align_up(local_mem_size, 16), 0);
  }

  std::vector<NS::SharedPtr<MTL::Buffer>> buffers_out;
  if (!arg_buffer_used) {
    encode_arguments(encoder, device, allocator, args, arg_sizes, num_args, is_pointer_arg);
  } else {
    encode_arguments_argbuffer(
      encoder, device, allocator, function.get(), args, arg_sizes, num_args, is_pointer_arg, buffers_out);
  }
  encoder->setBytes(&user_local_mem_size, sizeof(uint32_t), arg_buffer_used ? 1 : num_args);

  MTL::Size num_groups_size = MTL::Size::Make(
    num_groups[0],
    num_groups[1],
    num_groups[2]
  );

  MTL::Size threadgroup_size = MTL::Size::Make(
    group_size[0],
    group_size[1],
    group_size[2]
  );

  HIPSYCL_DEBUG_INFO << "Dispatching kernel '" << kernel_name << "' with grid size ("
                     << num_groups_size.width << ", " << num_groups_size.height << ", "
                     << num_groups_size.depth << ") and threadgroup size ("
                     << threadgroup_size.width << ", " << threadgroup_size.height
                     << ", " << threadgroup_size.depth << ")" << std::endl;

  encoder->dispatchThreadgroups(num_groups_size, threadgroup_size);

  encoder->endEncoding();

  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  if (command_buffer->error()) {
    NS::Error* err = command_buffer->error();
    std::string msg = "metal: Command buffer failed: ";
    if (err->localizedDescription()) {
      msg += err->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{msg});
  }

  HIPSYCL_DEBUG_INFO << "metal: Kernel '" << kernel_name
                     << "' executed successfully" << std::endl;

  return make_success();
}

result memset_device(
  MTL::CommandBuffer* command_buffer,
  metal_allocator* allocator,
  void* ptr,
  unsigned char pattern,
  std::size_t num_bytes)
{
  auto [buffer, offset, _3] = allocator->get_usm_block(ptr);
  if (!buffer) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: Failed to resolve USM pointer for memset"});
  }

  MTL::BlitCommandEncoder* blit_encoder = command_buffer->blitCommandEncoder();
  if (!blit_encoder) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: Failed to create blit encoder for memset"});
  }

  blit_encoder->fillBuffer(buffer, NS::Range::Make(offset, num_bytes), pattern);
  blit_encoder->endEncoding();

  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  if (command_buffer->error()) {
    NS::Error* err = command_buffer->error();
    std::string msg = "metal_queue: Memset failed: ";
    if (err->localizedDescription()) {
      msg += err->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{msg});
  }

  return make_success();
}

} // anonymous namespace

metal_inorder_queue::metal_inorder_queue(MTL::Device* device, metal_allocator* allocator, const device_id& id)
  : _device{device}, _allocator{allocator}, _device_id{id}
  , _sscp_code_object_invoker(this)
  , _kernel_cache{kernel_cache::get()}
{ }

std::shared_ptr<dag_node_event> metal_inorder_queue::insert_event() {
  HIPSYCL_DEBUG_INFO << "metal_queue: Inserting event into queue..." << std::endl;

  auto evt = std::make_shared<metal_node_event>();
  auto signal_channel = evt->get_signal_channel();

  // Schedule a task that will signal the event when all previous work completes
  _worker([signal_channel]() {
    signal_channel->signal();
  });

  return evt;
}

std::shared_ptr<dag_node_event> metal_inorder_queue::create_queue_completion_event() {
  // TODO: Metal backend doesn't support coarse-grained events yet
  return insert_event();
}

result metal_inorder_queue::submit_memcpy(memcpy_operation& op, const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "metal_queue: Submitting memcpy..." << std::endl;

  assert(op.source().get_base_ptr());
  assert(op.dest().get_base_ptr());

  void* src_ptr = op.source().get_base_ptr();
  void* dst_ptr = op.dest().get_base_ptr();
  std::size_t num_bytes = op.get_num_transferred_bytes();

  auto [src_buffer, _1, src_alloc_type] = _allocator->get_usm_block(src_ptr);
  auto [dst_buffer, _2, dst_alloc_type] = _allocator->get_usm_block(dst_ptr);
  bool src_is_device = src_buffer != nullptr; // treat shared USM as device memory for memcpy purposes
  bool dst_is_device = dst_buffer != nullptr;

  range<3> transferred_range = op.get_num_transferred_elements();
  range<3> src_allocation_shape = op.source().get_allocation_shape();
  range<3> dest_allocation_shape = op.dest().get_allocation_shape();
  id<3> src_offset = op.source().get_access_offset();
  id<3> dest_offset = op.dest().get_access_offset();
  std::size_t src_element_size = op.source().get_element_size();
  std::size_t dest_element_size = op.dest().get_element_size();

  auto linear_index = [](id<3> id, range<3> allocation_shape) {
    return id[2] + allocation_shape[2] * id[1] +
      allocation_shape[2] * allocation_shape[1] * id[0];
  };

  if (src_element_size != dest_element_size) {
    return register_error(
      __acpp_here(),
      error_info{"metal_queue: Source and destination element sizes do not match.",
                 error_type::invalid_parameter_error});
  }

  if (!src_is_device && !dst_is_device) {
    _worker([=]() {
      for (std::size_t surface = 0; surface < transferred_range[0]; ++surface) {
        for (std::size_t row = 0; row < transferred_range[1]; ++row) {
          id<3> src = src_offset; src[0] += surface; src[1] += row;
          id<3> dst = dest_offset; dst[0] += surface; dst[1] += row;

          const char* src_byte_ptr = (const char*)src_ptr +
            linear_index(src, src_allocation_shape) * src_element_size;
          char* dst_byte_ptr = (char*)dst_ptr +
            linear_index(dst, dest_allocation_shape) * dest_element_size;
          memcpy(dst_byte_ptr, src_byte_ptr, transferred_range[2] * src_element_size);
        }
      }
    });
    return make_success();
  }

  _worker([=]() {
    NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    NS::SharedPtr<MTL::CommandQueue> command_queue = NS::TransferPtr(_device->newCommandQueue());
    if (!command_queue) {
      register_error(make_error(__acpp_here(),
        error_info{"metal_queue: Failed to create command queue for memcpy"}));
      return;
    }

    MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
    if (!command_buffer) {
      register_error(make_error(__acpp_here(),
        error_info{"metal_queue: Failed to create command buffer for memcpy"}));
      return;
    }
    MTL::BlitCommandEncoder* blit_encoder = command_buffer->blitCommandEncoder();
    if (!blit_encoder) {
      register_error(make_error(__acpp_here(),
        error_info{"metal_queue: Failed to create blit encoder"}));
      return;
    }

    NS::SharedPtr<MTL::Buffer> temp_buffer;
    if (!src_is_device || !dst_is_device) {
      temp_buffer = NS::TransferPtr(
        _device->newBuffer(num_bytes, MTL::ResourceStorageModeShared));
      if (!temp_buffer) {
        register_error(make_error(__acpp_here(),
          error_info{"metal_queue: Failed to allocate temporary buffer"}));
        return;
      }
    }
    result res = make_success();

    MTL::Buffer* from;
    MTL::Buffer* to;
    size_t from_offset = 0;;
    size_t to_offset = 0;

    auto src_staging_offset = src_offset;;
    auto src_staging_shape = src_allocation_shape;

    if (src_is_device) {
      auto [src_buffer, src_offset, _] = _allocator->get_usm_block(src_ptr);
      from = src_buffer;
      from_offset = src_offset;
      if (!src_buffer) {
        register_error(make_error(__acpp_here(),
          error_info{"metal_queue: Failed to resolve source USM pointer"}));
        return;
      }
    } else {
      from = temp_buffer.get();
      // copy to staging buffer
      for (std::size_t surface = 0; surface < transferred_range[0]; ++surface) {
        for (std::size_t row = 0; row < transferred_range[1]; ++row) {
          id<3> src = src_offset; src[0] += surface; src[1] += row;
          id<3> dst = {surface, row, 0};

          const char* src_byte_ptr = (const char*)src_ptr +
            linear_index(src, src_allocation_shape) * src_element_size;
          char* dst_byte_ptr = (char*)temp_buffer->contents() +
            linear_index(dst, transferred_range) * src_element_size;
          memcpy(dst_byte_ptr, src_byte_ptr, transferred_range[2] * src_element_size);
        }
      }

      src_staging_offset = id<3>{0,0,0};
      src_staging_shape = transferred_range;
    }

    auto dst_staging_offset = dest_offset;
    auto dst_staging_shape = dest_allocation_shape;

    if (dst_is_device) {
      auto [dst_buffer, dst_offset, _] = _allocator->get_usm_block(dst_ptr);
      to = dst_buffer;
      to_offset = dst_offset;
      if (!dst_buffer) {
        register_error(make_error(__acpp_here(),
          error_info{"metal_queue: Failed to resolve destination USM pointer"}));
        return;
      }
    } else {
      // destination staging buffer
      to = temp_buffer.get();
      dst_staging_offset = {0,0,0};
      dst_staging_shape = transferred_range;
    }

    // copy from src or staging buffer to dst or staging buffer
    for (std::size_t surface = 0; surface < transferred_range[0]; ++surface) {
      for (std::size_t row = 0; row < transferred_range[1]; ++row) {
        id<3> src = src_staging_offset; src[0] += surface; src[1] += row;
        id<3> dst = dst_staging_offset; dst[0] += surface; dst[1] += row;

        size_t src_linear_index = linear_index(src, src_staging_shape);
        size_t dst_linear_index = linear_index(dst, dst_staging_shape);

        blit_encoder->copyFromBuffer(
          from,
          from_offset + src_linear_index * src_element_size,
          to,
          to_offset + dst_linear_index * dest_element_size,
          transferred_range[2] * src_element_size);
      }
    }

    blit_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (command_buffer->error()) {
      NS::Error* err = command_buffer->error();
      std::string msg = "metal_queue: Memcpy failed: ";
      if (err->localizedDescription()) {
        msg += err->localizedDescription()->utf8String();
      }
      register_error(make_error(__acpp_here(), error_info{msg}));
      return;
    }

    if (!dst_is_device) {
      // copy from staging buffer to host
      for (std::size_t surface = 0; surface < transferred_range[0]; ++surface) {
        for (std::size_t row = 0; row < transferred_range[1]; ++row) {
          id<3> src = dst_staging_offset; src[0] += surface; src[1] += row;
          id<3> dst = dest_offset; dst[0] += surface; dst[1] += row;

          const char* src_byte_ptr = (const char*)temp_buffer->contents() +
            linear_index(src, dst_staging_shape) * dest_element_size;
          char* dst_byte_ptr = (char*)dst_ptr +
            linear_index(dst, dest_allocation_shape) * dest_element_size;
          memcpy(dst_byte_ptr, src_byte_ptr, transferred_range[2] * dest_element_size);
        }
      }
    }

    if (!res.is_success()) {
      register_error(res);
    }
  });

  return make_success();
}

result metal_inorder_queue::submit_kernel(kernel_operation& op, const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "metal_queue: Submitting kernel..." << std::endl;

  rt::backend_kernel_launch_capabilities cap;
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);

  _worker([=, &op]() {
    NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto *node_ptr = node.get();
    result res = op.get_launcher().invoke(backend_id::metal, this, cap, node_ptr);
    if(!res.is_success()) {
      register_error(res);
    }
  });

  return make_success();
}

result metal_inorder_queue::submit_prefetch(prefetch_operation &, const dag_node_ptr&) {
  return make_success();
}

result metal_inorder_queue::submit_memset(memset_operation& op, const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "metal_queue: Submitting memset..." << std::endl;

  void* ptr = op.get_pointer();
  unsigned char pattern = op.get_pattern();
  std::size_t num_bytes = op.get_num_bytes();

  _worker([=]() {
    NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    NS::SharedPtr<MTL::CommandQueue> command_queue = NS::TransferPtr(_device->newCommandQueue());
    if (!command_queue) {
      register_error(make_error(__acpp_here(),
        error_info{"metal_queue: Failed to create command queue for memset"}));
      return;
    }

    MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
    if (!command_buffer) {
      register_error(make_error(__acpp_here(),
        error_info{"metal_queue: Failed to create command buffer for memset"}));
      return;
    }

    result res = memset_device(command_buffer, _allocator, ptr, pattern, num_bytes);

    if (!res.is_success()) {
      register_error(res);
    }
  });

  return make_success();
}

result metal_inorder_queue::submit_queue_wait_for(const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "metal_queue: Submitting wait for other queue..." << std::endl;

  auto evt = node->get_event();
  if (!evt) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: event is null"});
  }

  _worker([evt]() {
    evt->wait();
  });

  return make_success();
}

result metal_inorder_queue::submit_external_wait_for(const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "metal_queue: Submitting wait for external node..." << std::endl;

  if (!node) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: dag_node_ptr is null"});
  }

  _worker([node]() {
    node->wait();
  });

  return make_success();
}

result metal_inorder_queue::wait() {
  HIPSYCL_DEBUG_INFO << "metal_queue: Waiting for queue completion..." << std::endl;
  _worker.wait();
  return make_success();
}

device_id metal_inorder_queue::get_device() const {
  return _device_id;
}
void* metal_inorder_queue::get_native_type() const {
  return nullptr; // TODO
}

result metal_inorder_queue::query_status(inorder_queue_status& status) {
  status = inorder_queue_status{_worker.queue_size() == 0};
  return make_success();
}

result metal_inorder_queue::submit_sscp_kernel_from_code_object(hcf_object_id hcf_object,
  std::string_view kernel_name, const rt::hcf_kernel_info *kernel_info,
  const rt::range<3> &num_groups, const rt::range<3> &group_size,
  unsigned local_mem_size, void **args, std::size_t *arg_sizes,
  std::size_t num_args, const kernel_configuration &initial_config)
{
#ifdef HIPSYCL_WITH_SSCP_COMPILER
  HIPSYCL_DEBUG_INFO << "[Metal] submit_sscp_kernel_from_code_object() called for kernel: "
                     << kernel_name << std::endl;

  common::spin_lock_guard lock{_sscp_submission_spin_lock};

  // Validate kernel info
  if (!kernel_info) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: Could not obtain hcf kernel info for kernel " +
                 std::string{kernel_name}});
  }

  // Map C++ arguments to kernel arguments
  _arg_mapper.construct_mapping(*kernel_info, args, arg_sizes, num_args);

  if (!_arg_mapper.mapping_available()) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: Could not map C++ arguments to kernel arguments"});
  }

  // Set up adaptivity engine for image/kernel selection
  kernel_adaptivity_engine adaptivity_engine{
    hcf_object, kernel_name, kernel_info, _arg_mapper, num_groups,
    group_size, args, arg_sizes, num_args, local_mem_size};

  // Configure kernel compilation settings
  _config = initial_config;
  _config.append_base_configuration(
    kernel_base_config_parameter::backend_id, backend_id::metal);
  _config.append_base_configuration(
    kernel_base_config_parameter::compilation_flow, compilation_flow::sscp);
  _config.append_base_configuration(
    kernel_base_config_parameter::hcf_object_id, hcf_object);

  // Apply compilation flags and options from kernel info
  for (const auto& flag : kernel_info->get_compilation_flags())
    _config.set_build_flag(flag);
  for (const auto& opt : kernel_info->get_compilation_options())
    _config.set_build_option(opt.first, opt.second);

  _config.set_build_option(kernel_build_option::metal_max_args_for_flat_mode, metal_max_args_for_flat_mode);

  // Generate configuration IDs for caching
  auto binary_configuration_id = adaptivity_engine.finalize_binary_configuration(_config);
  auto code_object_configuration_id = binary_configuration_id;
  kernel_configuration::extend_hash(
    code_object_configuration_id,
    kernel_base_config_parameter::runtime_device,
    static_cast<int>(_device_id.get_id()));

  // Helper to get image name and kernel names from adaptivity engine
  auto get_image_and_kernel_names =
    [&](std::vector<std::string>& contained_kernels) -> std::string {
    return adaptivity_engine.select_image_and_kernels(&contained_kernels);
  };

  // JIT compiler function - invoked on cache miss to compile LLVM IR to Metal shader
  auto jit_compiler = [&](std::string& compiled_image) -> bool {
    std::vector<std::string> kernel_names;
    std::string selected_image_name = get_image_and_kernel_names(kernel_names);

    HIPSYCL_DEBUG_INFO << "[Metal] JIT compiling kernels from image: "
                       << selected_image_name << std::endl;

    std::unique_ptr<compiler::LLVMToBackendTranslator> translator =
      compiler::createLLVMToMetalTranslator(kernel_names);

    bool enable_dead_arg_elimination = kernel_names.size() == 1;

    rt::result err = glue::jit::compile_and_store_stats(
      translator.get(), hcf_object, selected_image_name, _config,
      binary_configuration_id, _reflection_map, compiled_image,
      enable_dead_arg_elimination);

    if (!err.is_success()) {
      register_error(err);
      return false;
    }

    return true;
  };

  // Code object constructor - creates executable object from compiled shader
  auto code_object_constructor = [&](const std::string& metal_shader) -> code_object* {
    std::vector<std::string> kernel_names;
    get_image_and_kernel_names(kernel_names);

    std::string target_arch = "metal";

    metal_sscp_executable_object* exec_obj = new metal_sscp_executable_object{
      metal_shader, target_arch, hcf_object, kernel_names, _device, _config};

    result r = exec_obj->get_build_result();

    if (!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    HIPSYCL_DEBUG_INFO << "[Metal] Successfully compiled shader to Metal library: "
                       << exec_obj->get_library() << std::endl;

    bool has_dead_arg_elimination = kernel_names.size() == 1;
    glue::jit::load_jit_output_metadata(*exec_obj, has_dead_arg_elimination,
                      binary_configuration_id);

    return exec_obj;
  };

  const code_object* obj = _kernel_cache->get_or_construct_jit_code_object(
    code_object_configuration_id, binary_configuration_id,
    jit_compiler, code_object_constructor);

  if (!obj) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: Code object construction failed"});
  }

  const auto& jit_output_metadata = obj->get_jit_output_metadata();
  const auto& retained_indices = jit_output_metadata.kernel_retained_arguments_indices;
  if (retained_indices.has_value()) {
    _arg_mapper.apply_dead_argument_elimination_mask(retained_indices.value());
  }
  if (!jit_output_metadata.is_free_of_indirect_access) {
    HIPSYCL_DEBUG_WARNING
      << "metal_queue: kernel '" << kernel_name
      << "' may perform indirect memory access, which is currently not supported by the Metal backend and can lead to undefined behavior (e.g., invalid or out-of-bounds memory access)."
      << std::endl;
  }

  // Get the Metal library from the code object
  const metal_executable_object* metal_obj =
    static_cast<const metal_executable_object*>(obj);
  MTL::Library* library = metal_obj->get_library();

  if (!library) {
    return make_error(__acpp_here(),
      error_info{"metal_queue: Metal library is null"});
  }

  return launch_kernel_from_library(
    library, _device, _allocator, kernel_name,
    num_groups, group_size, local_mem_size,
    _arg_mapper.get_mapped_args(),
    const_cast<std::size_t*>(_arg_mapper.get_mapped_arg_sizes()),
    _arg_mapper.get_mapped_num_args(),
    kernel_info,
    retained_indices);

#else
  return make_error(__acpp_here(),
    error_info{"metal_queue: SSCP kernel launch was requested, but AdaptiveCpp "
               "was not built with Metal SSCP support."});
#endif
}

metal_inorder_queue::~metal_inorder_queue() {
  _worker.halt();
}

worker_thread& metal_inorder_queue::get_worker() {
  return _worker;
}

result metal_sscp_code_object_invoker::submit_kernel(
  const kernel_operation& op,
  hcf_object_id hcf_object,
  const rt::range<3> &num_groups,
  const rt::range<3> &group_size,
  unsigned local_mem_size, void **args,
  std::size_t *arg_sizes, std::size_t num_args,
  std::string_view kernel_name,
  const rt::hcf_kernel_info* kernel_info,
  const kernel_configuration& config)
{
  return _queue->submit_sscp_kernel_from_code_object(
    hcf_object, kernel_name, kernel_info,
    num_groups, group_size, local_mem_size,
    args, arg_sizes, num_args,
    config);
}

} // namespace rt
} // namespace hipsycl
