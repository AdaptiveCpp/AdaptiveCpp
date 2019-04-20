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

#ifndef HIPSYCL_QUEUE_SUBMISSION_HOOKS_HPP
#define HIPSYCL_QUEUE_SUBMISSION_HOOKS_HPP

#include <unordered_map>

namespace cl {
namespace sycl {

class handler;

namespace detail {

#ifdef HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE

class queue_submission_hooks
{
public:
  using hook = function_class<void (sycl::handler&)>;
  using hook_id = std::size_t;

  void run_hooks(sycl::handler& cgh) const
  {
    for(const auto& element : _hooks)
      element.second(cgh);
  }

  hook_id add(hook&& h)
  {
    hook_id new_id = static_cast<hook_id>(_hooks.size());
    _hooks[new_id] = h;
    return new_id;
  }

  void remove(hook_id id)
  {
    _hooks.erase(id);
  }

private:
  using hook_list_type = 
    std::unordered_map<hook_id, hook>;
  
  hook_list_type _hooks;
};

using queue_submission_hooks_ptr = shared_ptr_class<queue_submission_hooks>;

template<typename, int, access::mode, access::target>
class automatic_placeholder_requirement_impl;

#endif // HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE

}
}
}

#endif
