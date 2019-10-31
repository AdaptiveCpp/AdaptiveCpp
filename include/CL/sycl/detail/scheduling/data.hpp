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

#ifndef HIPSYCL_DATA_HPP
#define HIPSYCL_DATA_HPP

#include <vector>
#include <utility>
#include <algorithm>

#include "../../access.hpp"
#include "../../id.hpp"
#include "../../range.hpp"
#include "dag_node.hpp"
#include "device_id.hpp"

namespace cl {
namespace sycl {
namespace detail {


enum class data_state : char
{
  unitialized = 0,
  outdated,
  current
};


struct data_user
{
  dag_node_ptr user;
  access::mode mode;
  access::target target;
  sycl::id<3> offset;
  sycl::range<3> range;
};

class data_user_tracker
{
public:
  using user_iterator = std::vector<data_user>::iterator;
  using const_user_iterator = std::vector<data_user>::const_iterator;

  const std::vector<data_user>& get_users() const;

  user_iterator find_user(dag_node_ptr user);
  const_user_iterator find_user(dag_node_ptr user) const;

  user_iterator users_begin();
  const_user_iterator users_begin() const;

  user_iterator users_end();
  const_user_iterator users_end() const;

  void release_dead_users();
  void add_user(dag_node_ptr user, 
                access::mode mode, 
                access::target target, 
                sycl::id<3> offset, 
                sycl::range<3> range);
private:
  std::vector<data_user> _users;
};

template<class Memory_descriptor = void*>
class data_region
{
public:
  using page_range = std::pair<std::size_t, std::size_t>;

  data_region(std::size_t size, std::size_t page_size)
  : _size{size}, _page_size{page_size}, _most_recent_data_version{-1}
  {
    _num_pages = (_size + page_size - 1) / page_size;
  }

  bool has_allocation(const device_id& d) const
  {
    return find_allocation(d) != _allocations.end();
  }

  void add_allocation(const device_id& d, Memory_descriptor memory_context)
  {
    // Make sure that there isn't already an allocation on the given device
    assert(!has_allocation(d));

    _allocations.push_back(data_allocation{
      d, memory_context, std::vector<int>(_num_pages, -1)});
  }

  void remove_allocation(const device_id& d)
  {
    assert(has_allocation(d));
    _allocations.erase(find_allocation(d));
  }

  /// Converts an offset into the data buffer (in bytes) and the
  /// data length (in bytes) into an equivalent \c page_range.
  page_range get_page_range(size_t data_offset, std::size_t data_size) const
  {
    std::size_t page_begin = (data_offset + _page_size - 1) / _page_size;
    std::size_t num_pages = (data_size + _page_size - 1) / _page_size;
    return std::make_pair(page_begin, num_pages);
  }

  /// Marks an allocation range on a given device most recent
  void mark_range_current(const device_id& d,
      std::size_t data_offset,
      std::size_t data_size)
  {
    page_range pr = get_page_range(data_offset, data_size);
    std::size_t first_page = pr.first;
    std::size_t num_pages = pr.second;

    ++_most_recent_data_version;

    auto allocation = find_allocation(d);

    for(std::size_t i = 0; i < num_pages; ++i)
    {
      std::size_t page_index = first_page + i;
      assert(page_index < _num_pages);
      assert(page_index < allocation->data_version.size());
      allocation->data_version[page_index] = _most_recent_data_version;
    }
  }

  bool is_range_current(const device_id& d,
      std::size_t data_offset,
      std::size_t data_size) const
  {
    page_range pr = get_page_range(data_offset, data_size);
    std::size_t first_page = pr.first;
    std::size_t num_pages = pr.second;

    assert(has_allocation(d));
    auto allocation = find_allocation(d);
    for(std::size_t page_index = first_page;
        page_index < first_page+num_pages;
        ++page_index)
    {
      assert(page_index < allocation->data_version.size());
      assert(allocation->data_version[page_index] <= _most_recent_data_version);

      if(allocation->data_version[page_index] < _most_recent_data_version)
        return false;
    }
    return true;
  }

  std::size_t get_number_of_consecutive_current_pages(const device_id& d,
      std::size_t data_offset,
      std::size_t data_size) const
  {
    page_range pr = get_page_range(data_offset, data_size);
    std::size_t first_page = pr.first;
    std::size_t num_pages = pr.second;

    assert(has_allocation(d));
    auto allocation = find_allocation(d);

    // TBD
  }


  data_user_tracker& get_users()
  { return _user_tracker; }

  const data_user_tracker& get_users() const
  { return _user_tracker; }

private:

  struct data_allocation
  {
    device_id dev;
    Memory_descriptor memory;
    std::vector<int> data_version;
  };

  std::vector<data_allocation> _allocations;
  int _most_recent_data_version;

  typename std::vector<data_allocation>::iterator
  find_allocation(device_id dev)
  {
    return std::find_if(_allocations.begin(), _allocations.end(),
      [dev](const data_allocation& current){
        return current.dev == dev;
    });
  }

  typename std::vector<data_allocation>::const_iterator
  find_allocation(device_id dev) const
  {
    return std::find_if(_allocations.cbegin(), _allocations.cend(),
      [dev](const data_allocation& current){
        return current.dev == dev;
    });
  }

  std::size_t _page_size;
  std::size_t _num_pages;
  const std::size_t _size;

  data_user_tracker _user_tracker;
};

using buffer_data_region = data_region<void*>;



}
}
}

#endif
