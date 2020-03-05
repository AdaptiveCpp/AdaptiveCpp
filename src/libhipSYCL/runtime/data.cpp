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
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/operations.hpp"

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



range_store::range_store(sycl::range<3> size)
: _size{size}, _contained_data(size.size())
{}

void range_store::add(const rect& r)
{
  this->for_each_element_in_range(r, 
    [this](sycl::id<3>, data_state& s){
      
    s = data_state::available;
  });
}

void range_store::remove(const rect& r)
{
  this->for_each_element_in_range(r, 
    [this](sycl::id<3>, data_state& s){

    s = data_state::empty;
  });
}

sycl::range<3> range_store::get_size() const
{ return _size; }

void range_store::intersections_with(const rect& r, 
                                    data_state desired_state,
                                    std::vector<rect>& out) const
{
  out.clear();
  
  sycl::id<3> rect_begin = r.first;
  sycl::id<3> rect_max = 
    rect_begin + sycl::id<3>{r.second[0], r.second[1], r.second[2]};


  std::vector<data_state> visited_entries(_contained_data.size(), data_state::empty);

  this->for_each_element_in_range(r, [&](sycl::id<3> pos, const data_state& entry){
    size_t linear_pos = get_index(pos);
    // Look for a potential new rect, if
    // * the starting position is of the state desired by the user
    // * the start position isn't covered already by another rect
    //   that was found previously
    if(entry == desired_state && 
      visited_entries[linear_pos] == data_state::empty) {
        
      // Find the largest contiguous rect for which all entries
      // are both of \c desired_state and unvisited
      sycl::range<3> rect_size = 
        find_max_contiguous_rect_extent(pos, rect_max, 
          [this,&visited_entries,desired_state](size_t linear_pos){
            return this->_contained_data[linear_pos] == desired_state
              && visited_entries[linear_pos] == data_state::empty;
          });

      rect found_rect = std::make_pair(pos, rect_size);
      out.push_back(found_rect);

      // Mark all pages inside the rect as visited
      this->for_each_element_in_range(
        found_rect, 
        visited_entries,
        [&](sycl::id<3> pos, data_state& entry){

          entry = data_state::available;
      });
    }
  });
}

bool range_store::entire_range_equals(
    const rect& r, data_state desired_state) const
{
  for(size_t x = r.first[0]; x < r.second[0]+r.first[0]; ++x){
    for(size_t y = r.first[1]; y < r.second[1]+r.first[1]; ++y){
      for(size_t z = r.first[2]; z < r.second[2]+r.first[2]; ++z){

        sycl::id<3> idx{x,y,z};
        size_t pos = get_index(idx);
        if(_contained_data[pos] != desired_state)
          return false;
      }
    }
  }

  return true;
}

}
}
}