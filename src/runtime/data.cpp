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
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/allocator.hpp"

namespace hipsycl {
namespace rt {

data_user_tracker::data_user_tracker(const data_user_tracker& other){
  _users = other._users;
}

data_user_tracker::data_user_tracker(data_user_tracker&& other)
: _users{std::move(other._users)}
{
  _users = other._users;
}

data_user_tracker& 
data_user_tracker::operator=(data_user_tracker other){
  _users = other._users;
  return *this;
}


data_user_tracker& 
data_user_tracker::operator=(data_user_tracker&& other){
  _users = std::move(other._users);
  return *this;
}

const std::vector<data_user>
data_user_tracker::get_users() const
{ 
  std::lock_guard<std::mutex> lock{_lock};
  return _users;
}


bool data_user_tracker::has_user(dag_node_ptr user) const
{
  std::lock_guard<std::mutex> lock{_lock};
  return std::find_if(_users.begin(), _users.end(), [user](const data_user &u) {
           return u.user.lock() == user;
         }) != _users.end();
}

void data_user_tracker::release_dead_users()
{
  std::lock_guard<std::mutex> lock{_lock};
  _users.erase(std::remove_if(_users.begin(), _users.end(),
                              [](const data_user &user) -> bool {
                                auto u = user.user.lock();
                                if (!u)
                                  return true;
                                return u->is_known_complete();
                              }),
               _users.end());
}

range_store::range_store(range<3> size)
: _size{size}, _contained_data(size.size())
{}

void range_store::add(const rect& r)
{
  this->for_each_element_in_range(r,
    [](id<3>, data_state& s){
      s = data_state::available;
    });
}

void range_store::remove(const rect& r)
{
  this->for_each_element_in_range(r, 
    [](id<3>, data_state& s){
      s = data_state::empty;
    });
}

range<3> range_store::get_size() const
{ return _size; }

void range_store::intersections_with(const rect& r, 
                                    data_state desired_state,
                                    std::vector<rect>& out) const
{
  out.clear();
  
  id<3> rect_begin = r.first;
  id<3> rect_max = 
    rect_begin + id<3>{r.second[0], r.second[1], r.second[2]};


  std::vector<data_state> visited_entries(_contained_data.size(), data_state::empty);

  this->for_each_element_in_range(r, [&](id<3> pos, const data_state& entry){
    size_t linear_pos = get_index(pos);
    // Look for a potential new rect, if
    // * the starting position is of the state desired by the user
    // * the start position isn't covered already by another rect
    //   that was found previously
    if(entry == desired_state && 
      visited_entries[linear_pos] == data_state::empty) {
        
      // Find the largest contiguous rect for which all entries
      // are both of \c desired_state and unvisited
      range<3> rect_size = 
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
        [&](id<3> pos, data_state& entry){

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

        id<3> idx{x,y,z};
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
