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

#include <cassert>
#include <algorithm>
#include "CL/sycl/detail/scheduling/data.hpp"
#include "CL/sycl/detail/scheduling/operations.hpp"

namespace cl {
namespace sycl {
namespace detail {

const std::vector<data_user>& 
data_user_tracker::get_users() const
{ return _users; }

data_user_tracker::user_iterator 
data_user_tracker::find_user(dag_node_ptr user)
{
  return std::find_if(_users.begin(), _users.end(), 
    [user](const data_user& u){ return u.user == user;});
}

data_user_tracker::const_user_iterator 
data_user_tracker::find_user(dag_node_ptr user) const
{
  return std::find_if(_users.begin(), _users.end(), 
    [user](const data_user& u){ return u.user == user;});
}

data_user_tracker::user_iterator 
data_user_tracker::users_begin()
{ return _users.begin(); }

data_user_tracker::const_user_iterator
data_user_tracker::users_begin() const
{ return _users.begin(); }

data_user_tracker::user_iterator 
data_user_tracker::users_end()
{ return _users.end(); }

data_user_tracker::const_user_iterator 
data_user_tracker::users_end() const
{ return _users.end(); }

void data_user_tracker::release_dead_users()
{
  std::vector<data_user> new_users;
  for(const auto& user : _users)
    if(!user.user->is_complete())
      new_users.push_back(user);
  _users = new_users;
}

void data_user_tracker::add_user(
  dag_node_ptr user, 
  access::mode mode, 
  access::target target, 
  sycl::id<3> offset, 
  sycl::range<3> range)
{
  assert(find_user(user) == users_end());

  _users.push_back(data_user{user, mode, target, offset, range});
}

}
}
}