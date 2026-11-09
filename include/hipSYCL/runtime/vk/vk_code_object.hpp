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

#pragma once

#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

#include <string>
#include <unordered_map>

#ifndef _WIN32
#include "spirv/unified1/spirv.hpp"
#else
#include "spirv-headers/spirv.hpp"
#endif

#include <spirv-tools/libspirv.h>
#include <vulkan/vulkan_raii.hpp>

namespace hipsycl {
namespace rt {

class vk_queue;
class vk_hardware_context;
class vk_executable_object;
class vk_kernel_object;

class vk_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  vk_sscp_code_object_invoker(vk_queue *queue) : _queue{queue} {}

  ~vk_sscp_code_object_invoker() {}

  result submit_kernel(const kernel_operation &op, hcf_object_id hcf_object,
                       const rt::range<3> &num_groups,
                       const rt::range<3> &group_size, unsigned local_mem_size,
                       void **args, std::size_t *arg_sizes,
                       std::size_t num_args, std::string_view kernel_name,
                       const rt::hcf_kernel_info *kernel_info,
                       const kernel_configuration &config) override;

private:
  vk_queue *_queue;
};

// Wrapper for information related to uniform buffers and descriptors needed
// to pass kernel arguments, and can be reused for another kernel execution
// once a previous execution completes with arguments set using an instance.
class vk_kernel_uniform_descriptors {
public:
  // Creates uniform buffers backend by memory, and descriptor sets, but
  // only if they have not already been created.
  void init(vk_kernel_object *kern_obj);

  vk::DescriptorSet get_descriptor_set() const { return _desc_set; }

  void set_completion_val(vk::Semaphore semaphore, uint64_t v) {
    _semaphore = semaphore;
    _completion_val = v;
  }

  bool is_available(bool stall);

  void *map_memory(unsigned binding, unsigned offset, unsigned size) {
    vk::DeviceSize mem_offset = _offsets[binding] + offset;
    return _dev_mem.mapMemory(mem_offset, size);
  }

  vk::Buffer get_uniform_buffer(size_t i) { return _uniform_buffers[i]; }

  void unmap_memory() { _dev_mem.unmapMemory(); }

private:
  void create_uniform_backing_buffers();
  void create_descriptor_set();

  vk_kernel_object *_kern_obj{nullptr};

  std::vector<vk::raii::Buffer> _uniform_buffers;
  std::vector<vk::DeviceSize> _offsets;
  vk::raii::DeviceMemory _dev_mem{nullptr};
  vk::raii::DescriptorSet _desc_set{nullptr};

  vk::Semaphore _semaphore{nullptr};
  uint64_t _completion_val{0};
};

// Wrapper for an kernel instance for a specific workgroup size which can be run
// once it is bound to a command-buffer and has its arguments set.
class vk_kernel_pipeline {
public:
  vk_kernel_pipeline();
  vk_kernel_pipeline(vk_kernel_object *kern_obj,
                     const rt::range<3> &group_size);

  void set_args(vk::CommandBuffer &, vk_kernel_uniform_descriptors &,
                glue::jit::cxx_argument_mapper &arg_mapper);
  void bind(vk::CommandBuffer &, vk_kernel_uniform_descriptors &);

  const rt::range<3> &get_group_size() const { return _group_size; }

private:
  vk_kernel_object *_kern_obj;
  rt::range<3> _group_size;

  vk::raii::Pipeline _compute_pipeline;
  vk::raii::PipelineLayout _compute_pipeline_layout;
};

using vk_kernel_pipeline_sp = std::shared_ptr<vk_kernel_pipeline>;

enum class spv_arg_kind {
  invalid,
  pod_ubo,
  pod_pushconstant,
  pointer_ubo,
  pointer_pushconstant,
  local
};

struct spv_kernel_argument {
  uint32_t _pos;
  uint32_t _descriptor_set;
  uint32_t _binding;
  uint32_t _offset;
  uint32_t _size;
  spv_arg_kind _kind;

  bool is_push_constant() const;
  bool is_uniform() const;

  friend std::ostream &operator<<(std::ostream &stream,
                                  const spv_kernel_argument &arg);
};

// Wrapper holding information about an abstract kernel from an executable
// object and information about its arguments. In order to run it must be
// made concrete for a specific workgroup size via `vk_kernel_pipeline`.
class vk_kernel_object {
public:
  vk_kernel_object();
  vk_kernel_object(std::string name, vk_executable_object *exe_obj);

  vk_kernel_pipeline_sp create_pipeline(const rt::range<3> &group_size);
  void create_descriptor_pool();
  void create_descriptor_layout();

  const std::string &get_name() const { return _name; }
  vk_executable_object *get_exe_obj() const { return _exe_obj; }

  void add_spv_arg(spv_kernel_argument);
  const std::vector<spv_kernel_argument> &get_spv_args() const;

  struct UniformArg {
    unsigned binding;
    unsigned size;
  };
  const std::vector<UniformArg> &get_uniform_args() const {
    return _uniform_args;
  }
  vk_kernel_uniform_descriptors &create_kernel_descriptors();

  size_t get_push_constants_size() const;
  void set_reqd_wg_size(unsigned x, unsigned y, unsigned z);
  bool check_reqd_wg_size(const rt::range<3> &group_size) const;
  vk::DescriptorSetLayout get_descriptor_set_layout() {
    return *_desc_set_layout;
  }
  vk::DescriptorPool get_descriptor_pool() { return *_desc_pool; }

  // Maximum number of concurrent invocations of a kernel
  static constexpr uint32_t MAX_INSTANCES = 10;

private:
  vk_executable_object *_exe_obj;
  std::string _name;
  std::unordered_map<rt::range<3>, vk_kernel_pipeline_sp, rt::range_hash>
      _pipelines;
  std::vector<size_t> _reqd_wg_size;
  std::vector<spv_kernel_argument> _args;
  std::vector<UniformArg> _uniform_args;

  vk::raii::DescriptorPool _desc_pool;
  vk::raii::DescriptorSetLayout _desc_set_layout;
  std::array<vk_kernel_uniform_descriptors, MAX_INSTANCES> _uniform_descriptors;
};

struct spv_reflection_data {
  uint32_t _uint_id = 0; // We only need to care about parsing uint constants
  std::unordered_map<uint32_t, uint32_t> _constants;
  std::unordered_map<uint32_t, std::string> _strings;
  vk_executable_object *_code_obj;
  std::vector<spv::Capability> _caps;
};

class vk_executable_object : public code_object {
public:
  vk_executable_object(vk_hardware_context *hw_ctx, hcf_object_id source,
                       const std::string &code_image,
                       const kernel_configuration &config);
  ~vk_executable_object();

  result get_build_result() const;

  code_object_state state() const override;
  code_format format() const override;
  backend_id managing_backend() const override;
  hcf_object_id hcf_source() const override;
  std::string target_arch() const override;

  vk::raii::ShaderModule &get_shader_module();
  const vk::raii::Device &get_device() const;
  vk_hardware_context *get_hw_ctx();

  std::vector<std::string> supported_backend_kernel_names() const override;

  bool contains(const std::string &backend_kernel_name) const override;

  compilation_flow source_compilation_flow() const override;
  kernel_configuration::id_type configuration_id() const override;

  // Only works if the module has been built successfully
  result get_kernel(std::string_view name, vk_kernel_object *&out) const;
  void add_kernel_name(const std::string &kernel);
  void add_kernel_handle(const std::string &kernel);
  void add_kernel_arg(const std::string &kernel, spv_kernel_argument arg);
  void set_kernel_reqd_wg_size(const std::string &kernel, unsigned, unsigned,
                               unsigned);
  void set_spec_const_wg_size(uint32_t, uint32_t, uint32_t);
  uint32_t *get_spec_const_wg_size();
  void add_subgroup_max_size_spec_constant(uint32_t spec_id);

  bool has_spec_const_subgroup_max_size() {
    return _spec_const_subgroup_max_size.has_value();
  }
  uint32_t get_spec_const_subgroup_max_size() {
    return _spec_const_subgroup_max_size.value();
  }

private:
  hcf_object_id _source;
  vk_hardware_context *_hw_ctx;
  spv_context _spv_ctx;

  result _build_status;
  kernel_configuration::id_type _id;

  vk::raii::ShaderModule _shader_module;

  std::optional<uint32_t> _spec_const_subgroup_max_size;
  std::vector<uint32_t> _spec_const_wg_size;
  std::vector<std::string> _kernel_names;
  // Mutable so that get_kernel() can return a non-const pointer a
  // vk_kernel_object in the map.
  mutable std::unordered_map<std::string_view, vk_kernel_object>
      _kernel_handles;
};

} // namespace rt
} // namespace hipsycl
