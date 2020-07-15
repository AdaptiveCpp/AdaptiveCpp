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

#include <limits>
#include <mutex>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>

#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/range.hpp"
#include "dag_node.hpp"
#include "device_id.hpp"

namespace hipsycl {
namespace rt {

void generic_pointer_free(device_id, void*);

class range_store
{
public:
  using rect = std::pair<sycl::id<3>,sycl::range<3>>;

  enum class data_state : char
  {
    empty = 0,
    available,
  };

  range_store(sycl::range<3> size);

  void add(const rect& r);
  
  void remove(const rect& r);

  sycl::range<3> get_size() const;

  void intersections_with(const rect& r, 
                          data_state desired_state,
                          std::vector<rect>& out) const;

  void intersections_with(const rect& r, 
                          std::vector<rect>& out) const
  { intersections_with(r, data_state::available, out); }

  void inverted_intersections_with(const rect& r, 
                                  std::vector<rect>& out) const
  { intersections_with(r, data_state::empty, out); }                             
    
  bool entire_range_equals(const rect&, data_state desired_state) const;

  bool entire_range_filled(const rect& r) const
  { return entire_range_equals(r, data_state::available); }

  bool entire_range_empty(const rect& r) const
  { return entire_range_equals(r, data_state::empty); }

private:
  template<class Entry_selection_predicate>
  sycl::range<3> find_max_contiguous_rect_extent(
    sycl::id<3> begin,
    sycl::id<3> search_range_end,
    Entry_selection_predicate p) const
  {
    // Find out length of longest contiguous row
    size_t z_size = 
      find_z_size(begin, search_range_end, p);

    // Now go to 2D and find out width of surface
    size_t y_size = 1;
    if(_size[1] > 1)
      y_size = find_y_size(begin, search_range_end, z_size, p);

    // Now 3D
    size_t x_size = 1;
    if(_size[0] > 1)
      x_size = find_x_size(begin, search_range_end, z_size, y_size, p);

    return sycl::range<3>{x_size, y_size, z_size};
    
  }

  template<class Entry_selection_predicate>
  size_t find_x_size(sycl::id<3> begin,
                     sycl::id<3> search_range_end,
                     size_t z_size,
                     size_t y_size,
                     Entry_selection_predicate p) const
  {
    for(size_t x_offset = 0; x_offset < search_range_end[0]-begin[0]; ++x_offset){
      sycl::id<3> surface_begin = begin;
      surface_begin[0] += x_offset;

      if(find_y_size(surface_begin, search_range_end, z_size, p) != y_size){
        return x_offset;
      }
    }
    return search_range_end[0] - begin[0];
  }

  template<class Entry_selection_predicate>
  size_t find_y_size(sycl::id<3> begin,
                    sycl::id<3> search_range_end,
                    size_t z_size,
                    Entry_selection_predicate p) const
  {
    for(size_t y_offset = 0; y_offset < search_range_end[1]-begin[1]; ++y_offset){
      sycl::id<3> row_begin = begin;
      row_begin[1] += y_offset;

      if(find_z_size(row_begin, search_range_end, p) != z_size){
        return y_offset;
      }
    }
    return search_range_end[1] - begin[1];
  }

  template<class Entry_selection_predicate>
  size_t find_z_size(sycl::id<3> begin,
                    sycl::id<3> search_range_end,
                    Entry_selection_predicate p) const
  {
    size_t base_pos = get_index(begin);
    size_t max_length = search_range_end[2] - begin[2];
    for(size_t offset = 0; offset < max_length; ++offset)
    {
      if(!p(base_pos + offset))
        return offset;
    }
    return max_length;
  }

  template<class Function>
  void for_each_element_in_range(const rect& r,
    std::vector<data_state>& v,
    Function f) const
  {
    for(size_t x = r.first[0]; x < r.second[0]+r.first[0]; ++x){
      for(size_t y = r.first[1]; y < r.second[1]+r.first[1]; ++y){
        for(size_t z = r.first[2]; z < r.second[2]+r.first[2]; ++z){

          sycl::id<3> idx{x,y,z};
          size_t pos = get_index(idx);
          assert(pos < v.size());
          f(sycl::id<3>{x,y,z}, v[pos]);
        }
      }
    }
  }

  template<class Function>
  void for_each_element_in_range(const rect& r, 
    const std::vector<data_state>& v, 
    Function f) const
  {
    for(size_t x = r.first[0]; x < r.second[0]+r.first[0]; ++x){
      for(size_t y = r.first[1]; y < r.second[1]+r.first[1]; ++y){
        for(size_t z = r.first[2]; z < r.second[2]+r.first[2]; ++z){

          sycl::id<3> idx{x,y,z};
          size_t pos = get_index(idx);
          assert(pos < v.size());
          f(sycl::id<3>{x,y,z}, v[pos]);
        }
      }
    }
  }

  template<class Function>
  void for_each_element_in_range(const rect& r, Function f) const
  { for_each_element_in_range(r, _contained_data, f); }

  template<class Function>
  void for_each_element_in_range(const rect& r, Function f)
  { for_each_element_in_range(r, _contained_data, f); }

  size_t get_index(sycl::id<3> pos) const
  {
    return pos[0] * _size[1] * _size[2] + pos[1] * _size[2] + pos[2]; 
  }

  std::vector<data_state> _contained_data;
  sycl::range<3> _size;
};


struct data_user
{
  dag_node_ptr user;
  sycl::access::mode mode;
  sycl::access::target target;
  sycl::id<3> offset;
  sycl::range<3> range;
};


class data_user_tracker
{
public:
  using user_iterator = std::vector<data_user>::iterator;
  using const_user_iterator = std::vector<data_user>::const_iterator;

  data_user_tracker() = default;
  data_user_tracker(const data_user_tracker& other);
  data_user_tracker(data_user_tracker&& other);
  data_user_tracker& operator=(data_user_tracker other);
  data_user_tracker& operator=(data_user_tracker&& other);

  const std::vector<data_user> get_users() const;

  template<class F>
  void for_each_user(F f){
    std::lock_guard<std::mutex> lock{_lock};
    for(auto& user : _users) {
      f(user);
    }
  }

  bool has_user(dag_node_ptr user) const;

  void release_dead_users();
  void add_user(dag_node_ptr user, 
                sycl::access::mode mode, 
                sycl::access::target target, 
                sycl::id<3> offset, 
                sycl::range<3> range);
private:
  std::vector<data_user> _users;
  mutable std::mutex _lock;
};

/// Manages data regions on different devices under
/// the assumptions:
/// * different devices may have copies of the same data regions
/// * data is managed at a "page" granularity (a 3D subrange of the data region)
/// * data accesses cover a 3D (sub-)range of the whole buffer
/// * data ranges on some devices may be outdated due to writes on
///   other devices
///
/// The interface of this class works in terms of numbers of elements, not
/// bytes!
template<class Memory_descriptor = void*>
class data_region
{
public:
  using page_range = std::pair<sycl::id<3>, sycl::range<3>>;
  using allocation_function = std::function<Memory_descriptor(
      sycl::range<3> num_elements, std::size_t element_size)>;

  using destruction_handler = std::function<void(data_region*)>;

  /// Construct object
  /// \param num_elements The 3D number of elements in each dimension. Each
  /// dimension must be a multiple of the page size
  /// \param page_size The size (numbers of elements) of the granularity of data
  /// management
  data_region(sycl::range<3> num_elements, std::size_t element_size,
              std::size_t page_size, destruction_handler on_destruction)
      : _element_size{element_size}, _num_elements{num_elements},
        _page_size{page_size}, _on_destruction{on_destruction} {

    unset_id();

    assert(page_size > 0);

    for(int i = 0; i < 3; ++i)
      assert(num_elements[i] % page_size == 0);
    
    _num_pages = num_elements / page_size;
  }


  ~data_region() {
    _on_destruction(this);
    for(const auto& alloc : _allocations) {
      if(alloc.memory && alloc.is_owned) {
        device_id dev = alloc.dev;
        generic_pointer_free(dev, alloc.memory);
      }
    }
  }

  bool has_allocation(const device_id& d) const
  {
    return find_allocation(d) != _allocations.end();
  }

  void add_placeholder_allocation(const device_id &d,  allocation_function f)
  {
    assert(!has_allocation(d));

    this->add_empty_allocation(d, nullptr, true);

    auto alloc = this->find_allocation(d);
    alloc->delayed_allocator = f;
  }

  void add_empty_allocation(const device_id &d,
                            Memory_descriptor memory_context,
                            bool takes_ownership = true) {
    // Make sure that there isn't already an allocation on the given device
    assert(!has_allocation(d));

    _allocations.push_back(data_allocation{
      d, memory_context, range_store{_num_pages}, takes_ownership});
    _allocations.back().invalid_pages.add(
        std::make_pair(sycl::id<3>{0,0,0},_num_pages));
  }

  void add_nonempty_allocation(const device_id &d,
                               Memory_descriptor memory_context,
                               bool takes_ownership = false) {
    // Make sure that there isn't already an allocation on the given device
    assert(!has_allocation(d));

    _allocations.push_back(data_allocation{
      d, memory_context, range_store{_num_pages}, takes_ownership});
    _allocations.back().invalid_pages.remove(
        std::make_pair(sycl::id<3>{0,0,0},_num_pages));
  }

  void remove_allocation(const device_id& d)
  {
    assert(has_allocation(d));
    _allocations.erase(find_allocation(d));
  }

  /// Converts an offset into the data buffer (in element numbers) and the
  /// data length (in element numbers) into an equivalent \c page_range.
  page_range get_page_range(sycl::id<3> data_offset,
                            sycl::range<3> data_range) const
  {
    sycl::id<3> page_begin{0,0,0};
    
    for(int i = 0; i < 3; ++i)
      page_begin[i] =  data_offset[i] / _page_size;

    sycl::id<3> page_end = page_begin;

    for (int i = 0; i < 3; ++i)
      page_end[i] =
          (data_offset[i] + data_range[i] + _page_size - 1) / _page_size;
    

    sycl::range<3> page_range{1,1,1};
    for (int i = 0; i < 3; ++i)
      page_range[i] = page_end[i] - page_begin[i];

    return std::make_pair(page_begin, page_range);
  }

  /// Marks an allocation range on a give device as not invalidated
  void mark_range_valid(const device_id &d, sycl::id<3> data_offset,
                        sycl::range<3> data_size)
  {
    page_range pr = get_page_range(data_offset, data_size);

    assert(has_allocation(d));

    auto alloc = find_allocation(d);
    alloc->invalid_pages.remove(pr);
  }
  
  /// Marks an allocation range on a given device most recent
  void mark_range_current(const device_id& d,
      sycl::id<3> data_offset,
      sycl::range<3> data_size)
  {
    page_range pr = get_page_range(data_offset, data_size);

    for(auto it = _allocations.begin(); 
      it != _allocations.end(); ++it){
      
      if(it->dev == d)
        it->invalid_pages.remove(pr);
      else
        it->invalid_pages.add(pr);
    }
  }

  void get_outdated_regions(const device_id& d,
                            sycl::id<3> data_offset,
                            sycl::range<3> data_size,
                            std::vector<range_store::rect>& out) const
  {
    assert(has_allocation(d));

    // Convert byte offsets/sizes to page ranges
    page_range pr = get_page_range(data_offset, data_size);
    sycl::id<3> first_page = pr.first;
    sycl::range<3> num_pages = pr.second;

    // Find outdated regions among pages
    auto allocation = find_allocation(d);
    allocation->invalid_pages.intersections_with(
      std::make_pair(first_page, num_pages), out);

    // Convert back to num elements
    for(range_store::rect& r : out) {
      for(int i = 0; i < 3; ++i) {
        r.first[i] *= _page_size;
        r.second[i] *= _page_size;
      }
    }
  }

  void get_update_source_candidates(
              const device_id& d,
              const range_store::rect& data_range,
              std::vector<std::pair<device_id, range_store::rect>>& update_sources) const
  {
    update_sources.clear();

    page_range pr = get_page_range(data_range.first, data_range.second);

    for(const auto& alloc : _allocations) {
      if(alloc.dev != d){
        if(alloc.invalid_pages.entire_range_empty(pr)) {
          update_sources.push_back(std::make_pair(alloc.dev, data_range));
        }
      }
    }
    if(update_sources.empty()){
      assert(false && "Could not find valid data source for updating data buffer - "
              "this can happen if several data transfers are required to update accessed range, "
              "which is not yet supported.");
    }
  }

  bool is_range_current(const device_id& d,
      sycl::id<3> data_offset,
      sycl::range<3> data_size) const
  {
    assert(has_allocation(d));
    auto allocation = find_allocation(d);

    return allocation->invalid_pages.entire_range_empty(
      std::make_pair(data_offset, data_size));
  }

  data_user_tracker& get_users()
  { return _user_tracker; }

  const data_user_tracker& get_users() const
  { return _user_tracker; }

  std::unique_ptr<data_region> create_fork() const
  {
    return std::make_unique<data_region>(*this);  
  }

  void apply_fork(data_region *fork)
  {
    fork->materialize_placeholder_allocations();
    *this = *fork;
  }

  /// Set an id for this data region. This is used by the \c dag_enumerator
  /// to efficiently identify this object during dag expansion/scheduling
  void set_enumerated_id(std::size_t id) { _enumerated_id = id; }
  std::size_t get_id() const { return _enumerated_id; }

  void unset_id()
  { _enumerated_id = std::numeric_limits<std::size_t>::max(); }

  bool has_id() const
  { return _enumerated_id != std::numeric_limits<std::size_t>::max(); }

  std::size_t get_element_size() const { return _element_size; }

  sycl::range<3> get_num_elements() const { return _num_elements; }

  Memory_descriptor get_memory(device_id dev) const
  {
    assert(has_allocation(dev));
    return find_allocation(dev)->memory;
  }
private:
  std::size_t _enumerated_id;
  std::size_t _element_size;

  struct data_allocation
  {
    device_id dev;
    Memory_descriptor memory;
    range_store invalid_pages;
    bool is_owned;
    
    allocation_function delayed_allocator;
  };

  std::vector<data_allocation> _allocations;
  destruction_handler _on_destruction;

  void materialize_placeholder_allocations()
  {
    for (data_allocation &alloc : _allocations) {
      if (alloc.memory == nullptr) {
        alloc.memory =
            alloc.delayed_allocator(_num_elements, _element_size);
        
        alloc.delayed_allocator = allocation_function{};
      }
    }  
  }
  
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
  sycl::range<3> _num_pages;
  sycl::range<3> _num_elements;

  data_user_tracker _user_tracker;
};

using buffer_data_region = data_region<void*>;



}
}

#endif
