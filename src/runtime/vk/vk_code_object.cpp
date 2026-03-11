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
#include "hipSYCL/runtime/vk/vk_code_object.hpp"
#include "hipSYCL/runtime/vk/vk_hardware_manager.hpp"
#include "hipSYCL/runtime/vk/vk_queue.hpp"

#include "spirv/unified1/NonSemanticClspvReflection.h"
#include "spirv/unified1/spirv.hpp"

namespace hipsycl {
namespace rt {

result vk_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, hcf_object_id hcf_object,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned int local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, std::string_view kernel_name,
    const rt::hcf_kernel_info *kernel_info,
    const kernel_configuration &config) {
  assert(_queue);
  return _queue->submit_sscp_kernel_from_code_object(
      hcf_object, kernel_name, kernel_info, num_groups, group_size,
      local_mem_size, args, arg_sizes, num_args, config);
}

vk_kernel_object::vk_kernel_object()
    : _exe_obj(nullptr), _name(), _uniform_arg_size(0),
      _desc_set_layout(nullptr), _desc_pool(nullptr) {}

vk_kernel_object::vk_kernel_object(std::string name,
                                   vk_executable_object *exe_obj)
    : _exe_obj(exe_obj), _name(name), _uniform_arg_size(0),
      _desc_set_layout(nullptr), _desc_pool(nullptr) {}

void vk_kernel_object::add_spv_arg(spv_kernel_argument arg) {
  _args.push_back(arg);
}

void vk_kernel_object::set_reqd_wg_size(unsigned x, unsigned y, unsigned z) {
  _reqd_wg_size = {x, y, z};
}

bool vk_kernel_object::check_reqd_wg_size(
    const rt::range<3> &group_size) const {
  if (_reqd_wg_size.empty())
    return true;

  return _reqd_wg_size[0] == group_size[0] &&
         _reqd_wg_size[1] == group_size[1] && _reqd_wg_size[2] == group_size[2];
}

const std::vector<spv_kernel_argument> &vk_kernel_object::get_spv_args() const {
  return _args;
}

void vk_kernel_object::create_descriptor_pool() {
  // All uniform args are condensed into a single buffer
  constexpr unsigned num_uniform_args = 1;
  std::vector<vk::DescriptorPoolSize> pool_sizes = {vk::DescriptorPoolSize(
      vk::DescriptorType::eUniformBuffer, num_uniform_args * MAX_INSTANCES)};

  vk::DescriptorPoolCreateInfo pool_info{};
  pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  pool_info.maxSets = MAX_INSTANCES;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();

  auto &device = _exe_obj->get_device();
  _desc_pool = vk::raii::DescriptorPool(device, pool_info);
}

void vk_kernel_object::create_descriptor_sets() {
  for (const auto &spv_arg : _args) {
    if (spv_arg.is_uniform()) {
      auto binding = spv_arg._binding;
      assert(binding == 0 && "only one binding currently supported");
      if (binding == 0) {
        _uniform_arg_size += spv_arg._size;
      } else {
        print_error(
            __acpp_here(),
            error_info{"Non zero uniform buffer binding is not supported"});
      }
    }
  }

  std::vector<vk::DescriptorSetLayoutBinding> layout_bindings{
      vk::DescriptorSetLayoutBinding(
          0 /* binding */, vk::DescriptorType::eUniformBuffer, 1,
          vk::ShaderStageFlagBits::eCompute, nullptr)};

  auto &device = _exe_obj->get_device();
  vk::DescriptorSetLayoutCreateInfo layout_info({}, layout_bindings);
  _desc_set_layout = vk::raii::DescriptorSetLayout(device, layout_info);

  vk::DescriptorSetAllocateInfo alloc_info(_desc_pool, *_desc_set_layout);
  _desc_sets = device.allocateDescriptorSets(alloc_info);
}

bool spv_kernel_argument::is_push_constant() const {
  return _kind == spv_arg_kind::pointer_pushconstant ||
         _kind == spv_arg_kind::pod_pushconstant;
}

bool spv_kernel_argument::is_uniform() const {
  return _kind == spv_arg_kind::pointer_ubo || _kind == spv_arg_kind::pod_ubo;
}

std::ostream &operator<<(std::ostream &stream, const spv_kernel_argument &arg) {
  stream << "index: " << arg._pos;
  stream << ", desc set: " << arg._descriptor_set;
  stream << ", binding: " << arg._binding;
  stream << ", offset: " << arg._offset;
  stream << ", size: " << arg._size;
  stream << ", type: ";
  switch (arg._kind) {
  case spv_arg_kind::invalid:
    stream << "invalid";
    break;
  case spv_arg_kind::pod_ubo:
    stream << "ubo pod";
    break;
  case spv_arg_kind::pod_pushconstant:
    stream << "pod push constant";
    break;
  case spv_arg_kind::pointer_ubo:
    stream << "ubo pointer";
    break;
  case spv_arg_kind::pointer_pushconstant:
    stream << "pointer push constant";
    break;
  case spv_arg_kind::local:
    stream << "local";
    break;
  }
  return stream;
}

size_t vk_kernel_object::get_push_constants_size() const {
  size_t max_size = 0;
  size_t index = 0;
  for (const auto &spv_arg : _args) {
    if (spv_arg.is_push_constant()) {
      if (spv_arg._pos >= index) {
        max_size = spv_arg._offset + spv_arg._size;
      }
    }
  }

  // VK requirement that push constants be 4-byte aligned
  max_size += max_size % 4;
  return max_size;
}

vk_kernel_pipeline_sp
vk_kernel_object::create_pipeline(const rt::range<3> &group_size) {
  auto it = _pipelines.find(group_size);
  if (it != _pipelines.end()) {
    return it->second;
  }

  vk_kernel_pipeline_sp pipeline =
      std::make_shared<vk_kernel_pipeline>(this, group_size);
  _pipelines.insert({group_size, pipeline});
  return pipeline;
}

vk_kernel_pipeline::vk_kernel_pipeline()
    : _kern_obj(nullptr), _group_size(), _compute_pipeline(nullptr),
      _compute_pipeline_layout(nullptr) {}

vk_kernel_pipeline::vk_kernel_pipeline(vk_kernel_object *kern_obj,
                                       const rt::range<3> &group_size)
    : _kern_obj(kern_obj), _group_size(group_size), _compute_pipeline(nullptr),
      _compute_pipeline_layout(nullptr) {
  if (!kern_obj->check_reqd_wg_size(group_size)) {
    print_error(__acpp_here(),
                error_info{"reqd wg size doesn't match enqueued kernel size"});
  }

  create_uniform_backing_buffers();
  create_compute_pipeline();
}

void vk_kernel_pipeline::create_uniform_backing_buffers() {
  if (unsigned bytes = _kern_obj->get_uniform_arg_size(); bytes > 0) {
    vk_allocator *allocator =
        _kern_obj->get_exe_obj()->get_hw_ctx()->get_allocator();
    auto [uniform_buffer_raii, uniform_mem_raii] = allocator->create_buffer(
        bytes, vk::BufferUsageFlagBits::eUniformBuffer);
    _uniform_buffer_raii = std::move(uniform_buffer_raii);
    _uniform_mem_raii = std::move(uniform_mem_raii);
  }
}

void vk_kernel_pipeline::create_compute_pipeline() {
  auto &device = _kern_obj->get_exe_obj()->get_device();
  auto desc_set_layout = _kern_obj->get_descriptor_set_layout();

  size_t push_constant_bytes = _kern_obj->get_push_constants_size();
  vk::PushConstantRange push_constant_ranges(vk::ShaderStageFlagBits::eCompute,
                                             0, push_constant_bytes);
  HIPSYCL_DEBUG_INFO
      << "vk_kernel_pipeline: Created compute pipeline layout with "
      << push_constant_bytes << " byte push constant range" << std::endl;

  vk::PipelineLayoutCreateInfo pipeline_layout_info =
      push_constant_bytes ? vk::PipelineLayoutCreateInfo({}, desc_set_layout,
                                                         push_constant_ranges)
                          : vk::PipelineLayoutCreateInfo({}, desc_set_layout);

  _compute_pipeline_layout =
      vk::raii::PipelineLayout(device, pipeline_layout_info);

  // Create compute pipeline with a single stage for the compute shader
  vk::PipelineShaderStageCreateInfo stage_info(
      {}, vk::ShaderStageFlagBits::eCompute,
      _kern_obj->get_exe_obj()->get_shader_module(),
      _kern_obj->get_name().c_str());

  std::vector<vk::SpecializationMapEntry> spec_map_entries;
  std::vector<uint32_t> spec_map_data;
  if (uint32_t *spec_const_wg =
          _kern_obj->get_exe_obj()->get_spec_const_wg_size()) {
    // Specialization constant for number of threads/invocations/work-items
    // in compute shader work-group.
    spec_map_entries.emplace_back(0, 0, sizeof(uint32_t));
    spec_map_data.push_back(spec_const_wg[0]);
    spec_map_entries.emplace_back(1, sizeof(uint32_t), sizeof(uint32_t));
    spec_map_data.push_back(spec_const_wg[1]);
    spec_map_entries.emplace_back(2, 2 * sizeof(uint32_t), sizeof(uint32_t));
    spec_map_data.push_back(spec_const_wg[2]);

    HIPSYCL_DEBUG_INFO << "Specialization constant set for work group size ("
                       << spec_const_wg[0] << ", " << spec_const_wg[1] << ","
                       << spec_const_wg[2] << ")\n";
  }

  if (_kern_obj->get_exe_obj()->has_spec_const_subgroup_max_size()) {
    size_t offset = spec_map_entries.size() * sizeof(uint32_t);
    auto spec_id = _kern_obj->get_exe_obj()->get_spec_const_subgroup_max_size();
    spec_map_entries.emplace_back(spec_id, offset, sizeof(uint32_t));
    auto device_subgroup_size =
        _kern_obj->get_exe_obj()->get_hw_ctx()->get_subgroup_size();
    spec_map_data.push_back(device_subgroup_size);

    HIPSYCL_DEBUG_INFO << "Specialization constant id " << spec_id
                       << " set for subgroup group size "
                       << device_subgroup_size << std::endl;
  }

  vk::SpecializationInfo spec_info(
      spec_map_entries.size(), spec_map_entries.data(),
      spec_map_entries.size() * sizeof(uint32_t), spec_map_data.data());
  stage_info.pSpecializationInfo =
      spec_map_entries.empty() ? nullptr : &spec_info;

  vk::ComputePipelineCreateInfo pipeline_info({}, stage_info,
                                              _compute_pipeline_layout);
  _compute_pipeline = vk::raii::Pipeline(device, nullptr, pipeline_info);
}

void vk_kernel_pipeline::set_args(vk::CommandBuffer &cmd_buf,
                                  glue::jit::cxx_argument_mapper &arg_mapper) {
  void **kernel_args = arg_mapper.get_mapped_args();

  size_t push_constant_bytes = _kern_obj->get_push_constants_size();
  std::vector<uint8_t> push_constants(push_constant_bytes, 0);

  for (const auto &spv_arg : _kern_obj->get_spv_args()) {
    void *kernel_arg = kernel_args[spv_arg._pos];
    if (spv_arg.is_push_constant()) {
      void *dest = push_constants.data() + spv_arg._offset;
      std::memcpy(dest, kernel_arg, spv_arg._size);
    } else if (spv_arg.is_uniform()) {
      void *vptr = _uniform_mem_raii.mapMemory(spv_arg._offset, spv_arg._size);
      std::memcpy(vptr, kernel_arg, spv_arg._size);
      _uniform_mem_raii.unmapMemory();
    } else {
      assert(false && "handle unknown arg");
    }
  }

  if (_kern_obj->get_uniform_arg_size() > 0) {
    std::vector<vk::WriteDescriptorSet> descriptor_writes;
    vk::DescriptorBufferInfo buffer_info(_uniform_buffer_raii, 0,
                                         VK_WHOLE_SIZE);

    constexpr unsigned binding = 0;
    vk::WriteDescriptorSet write_desc_set(
        _kern_obj->get_descriptor_set(), binding, 0,
        vk::DescriptorType::eUniformBuffer, {}, buffer_info);
    descriptor_writes.push_back(write_desc_set);

    auto &device = _kern_obj->get_exe_obj()->get_device();
    device.updateDescriptorSets(descriptor_writes, {});
  }

  if (push_constant_bytes) {
    cmd_buf.pushConstants<uint8_t>(_compute_pipeline_layout,
                                   vk::ShaderStageFlagBits::eCompute, 0,
                                   push_constants);
  }
}

void vk_kernel_pipeline::bind(vk::CommandBuffer &cmd_buf) {
  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, _compute_pipeline);
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             _compute_pipeline_layout, 0,
                             {_kern_obj->get_descriptor_set()}, {});
}

static spv_arg_kind get_spv_arg_kind(uint32_t inst) {
  switch (static_cast<NonSemanticClspvReflectionInstructions>(inst)) {
  case NonSemanticClspvReflectionArgumentPodUniform:
    return spv_arg_kind::pod_ubo;
  case NonSemanticClspvReflectionArgumentPodPushConstant:
    return spv_arg_kind::pod_pushconstant;
  case NonSemanticClspvReflectionArgumentPointerUniform:
    return spv_arg_kind::pointer_ubo;
  case NonSemanticClspvReflectionArgumentPointerPushConstant:
    return spv_arg_kind::pointer_pushconstant;
  case NonSemanticClspvReflectionArgumentWorkgroup:
    return spv_arg_kind::local;
  default:
    print_error(__acpp_here(),
                error_info{"failed to identity reflected instruction"});
    break;
  }
  assert(false && "unknown argument");
  return spv_arg_kind::invalid;
};

static spv_result_t parse_reflection(void *user_data,
                                     const spv_parsed_instruction_t *inst) {
  auto *parse_data = reinterpret_cast<spv_reflection_data *>(user_data);

  switch (inst->opcode) {
  case spv::OpTypeInt:
    if (inst->words[2] == 32 && inst->words[3] == 0) {
      parse_data->_id = inst->result_id;
    }
    break;
  case spv::OpConstant:
    if (inst->words[1] == parse_data->_id) {
      parse_data->_constants[inst->result_id] = inst->words[3];
    }
    break;
  case spv::OpString:
    parse_data->_strings[inst->result_id] =
        std::string(reinterpret_cast<const char *>(&inst->words[2]));
    break;

  case spv::OpExtInst:
    if (inst->ext_inst_type == SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION) {
      auto ext_inst = inst->words[4];
      switch (ext_inst) {
      case NonSemanticClspvReflectionKernel: {
        const auto &name = parse_data->_strings[inst->words[6]];
        parse_data->_strings[inst->result_id] = name;
        break;
      }
      case NonSemanticClspvReflectionArgumentInfo:
        break;
      case NonSemanticClspvReflectionPropertyRequiredWorkgroupSize: {
        auto kernel = parse_data->_strings[inst->words[5]];
        auto x = parse_data->_constants[inst->words[6]];
        auto y = parse_data->_constants[inst->words[7]];
        auto z = parse_data->_constants[inst->words[8]];
        parse_data->_code_obj->set_kernel_reqd_wg_size(kernel, x, y, z);
        break;
      }
      case NonSemanticClspvReflectionSpecConstantWorkgroupSize: {
        auto x = parse_data->_constants[inst->words[5]];
        auto y = parse_data->_constants[inst->words[6]];
        auto z = parse_data->_constants[inst->words[7]];
        parse_data->_code_obj->set_spec_const_wg_size(x, y, z);
        break;
      }
      case NonSemanticClspvReflectionArgumentPointerPushConstant:
      case NonSemanticClspvReflectionArgumentPodPushConstant: {
        auto kernel = parse_data->_strings[inst->words[5]];
        auto ordinal = parse_data->_constants[inst->words[6]];
        auto offset = parse_data->_constants[inst->words[7]];
        auto size = parse_data->_constants[inst->words[8]];

        spv_arg_kind kind = get_spv_arg_kind(ext_inst);
        spv_kernel_argument arg = {ordinal, 0, 0, offset, size, kind};
        parse_data->_code_obj->add_kernel_arg(kernel, std::move(arg));
        break;
      }
      case NonSemanticClspvReflectionArgumentPodUniform:
      case NonSemanticClspvReflectionArgumentPointerUniform: {
        auto kernel = parse_data->_strings[inst->words[5]];
        auto ordinal = parse_data->_constants[inst->words[6]];
        auto descriptor_set = parse_data->_constants[inst->words[7]];
        auto binding = parse_data->_constants[inst->words[8]];
        auto offset = parse_data->_constants[inst->words[9]];
        auto size = parse_data->_constants[inst->words[10]];
        auto kind = get_spv_arg_kind(ext_inst);
        spv_kernel_argument arg = {ordinal, descriptor_set, binding,
                                   offset,  size,           kind};
        parse_data->_code_obj->add_kernel_arg(kernel, std::move(arg));
        break;
      }
      case NonSemanticClspvReflectionWorkgroupVariableSize:
        // Contains size of local memory used, ignore
        break;
      case NonSemanticClspvReflectionSpecConstantSubgroupMaxSize: {
        auto size_id = parse_data->_constants[inst->words[5]];
        parse_data->_code_obj->add_subgroup_max_size_spec_constant(size_id);
        break;
      }
      default:
        assert(false && " Unexpected type");
        return SPV_ERROR_INVALID_DATA;
      }
    }
    break;
  default:
    break;
  }

  return SPV_SUCCESS;
}

vk_executable_object::vk_executable_object(
    vk_hardware_context *hw_ctx, hcf_object_id source,
    const std::string &code_image, const kernel_configuration &config,
    std::vector<std::string> kernel_names)
    : _source{source}, _hw_ctx(hw_ctx), _id{config.generate_id()},
      _shader_module(nullptr), _kernel_names(kernel_names) {
  _spv_ctx = spvContextCreate(SPV_ENV_VULKAN_1_3);
  if (_spv_ctx == nullptr)
    print_error(__acpp_here(), error_info{"failed to create spirv context"});

  vk::ShaderModuleCreateInfo create_info(
      {}, code_image.size() * sizeof(char),
      reinterpret_cast<const uint32_t *>(code_image.c_str()));
  _shader_module = _hw_ctx->get_device().createShaderModule(create_info);

  for (const std::string &kernel_name : _kernel_names) {
    _kernel_handles.insert({kernel_name, vk_kernel_object(kernel_name, this)});
  }

  spv_reflection_data reflection;
  reflection._code_obj = this;
  auto result =
      spvBinaryParse(_spv_ctx, &reflection,
                     reinterpret_cast<const uint32_t *>(code_image.c_str()),
                     code_image.size() / 4 /* SPIRV word size */, nullptr,
                     parse_reflection, nullptr);
  if (result != SPV_SUCCESS) {
    print_error(__acpp_here(), error_info{"failed to parse spirv"});
  }

  for (auto &kern : _kernel_handles) {
    kern.second.create_descriptor_pool();
    kern.second.create_descriptor_sets();
  }

  _build_status = make_success();
}

vk_executable_object::~vk_executable_object() { spvContextDestroy(_spv_ctx); }

void vk_executable_object::add_kernel_arg(const std::string &kernel,
                                          spv_kernel_argument arg) {
  auto it = _kernel_handles.find(kernel);
  if (it == _kernel_handles.end()) {
    print_error(__acpp_here(),
                error_info{"vk_executable_object: Unknown kernel name"});
    return;
  }
  auto &kernel_handle = it->second;
  kernel_handle.add_spv_arg(arg);

  HIPSYCL_DEBUG_INFO << "vk_executable_object: kernel " << kernel
                     << "added arg - " << arg << std::endl;
}

void vk_executable_object::set_kernel_reqd_wg_size(const std::string &kernel,
                                                   unsigned x, unsigned y,
                                                   unsigned z) {
  auto it = _kernel_handles.find(kernel);
  if (it == _kernel_handles.end()) {
    print_error(__acpp_here(),
                error_info{"vk_executable_object: Unknown kernel name"});
    return;
  }
  auto &kernel_handle = it->second;
  kernel_handle.set_reqd_wg_size(x, y, z);

  HIPSYCL_DEBUG_INFO << "kernel " << kernel << "required wg size (" << x << ","
                     << y << "," << z << ")\n";
}

void vk_executable_object::set_spec_const_wg_size(uint32_t x, uint32_t y,
                                                  uint32_t z) {
  _spec_const_wg_size = {x, y, z};
  HIPSYCL_DEBUG_INFO << "Spec constant wg size (" << x << "," << y << "," << z
                     << ")\n";
}

void vk_executable_object::add_subgroup_max_size_spec_constant(
    uint32_t spec_id) {
  _spec_const_subgroup_max_size = spec_id;
}

uint32_t *vk_executable_object::get_spec_const_wg_size() {
  if (_spec_const_wg_size.empty()) {
    return nullptr;
  }
  return _spec_const_wg_size.data();
}

result vk_executable_object::get_build_result() const { return _build_status; }

code_object_state vk_executable_object::state() const {
  if (_build_status.is_success())
    return code_object_state::executable;
  return code_object_state::invalid;
}

code_format vk_executable_object::format() const { return code_format::spirv; }

backend_id vk_executable_object::managing_backend() const {
  return backend_id::vk;
}

hcf_object_id vk_executable_object::hcf_source() const { return _source; }

std::string vk_executable_object::target_arch() const { return "spirv64"; }

std::vector<std::string>
vk_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

bool vk_executable_object::contains(
    const std::string &backend_kernel_name) const {
  return _kernel_handles.find(backend_kernel_name) != _kernel_handles.end();
}

compilation_flow vk_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

kernel_configuration::id_type vk_executable_object::configuration_id() const {
  return _id;
}

result vk_executable_object::get_kernel(std::string_view name,
                                        vk_kernel_object *&out) const {
  if (!_build_status.is_success())
    return _build_status;
  const auto &it = _kernel_handles.find(name);
  if (it == _kernel_handles.end())
    return make_error(__acpp_here(),
                      error_info{"vk_executable_object: Unknown kernel name"});
  out = &it->second;
  return make_success();
}

vk::raii::ShaderModule &vk_executable_object::get_shader_module() {
  return _shader_module;
}

const vk::raii::Device &vk_executable_object::get_device() const {
  return _hw_ctx->get_device();
}

vk_hardware_context *vk_executable_object::get_hw_ctx() { return _hw_ctx; }

} // namespace rt
} // namespace hipsycl
