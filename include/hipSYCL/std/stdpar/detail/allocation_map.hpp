/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#ifndef HIPSYCL_ALLOCATION_MAP_HPP
#define HIPSYCL_ALLOCATION_MAP_HPP


#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <atomic>
#include <algorithm>

extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void*);

namespace hipsycl::stdpar {

struct default_allocation_map_payload {};

template<class UserPayload = default_allocation_map_payload>
class allocation_map {
public:
  static_assert(sizeof(void*) == 8, "Unsupported pointer size");
  static_assert(std::is_trivial_v<UserPayload>, "UserPayload must be trivial type");

  allocation_map()
  : _num_in_progress_operations{0}, _num_empty_leaves{0} {}

  struct value_type : public UserPayload {
    std::size_t allocation_size;
  };

  // Access entry of allocation that address belongs to, or nullptr if the address
  // does not belong to a known allocation.
  value_type* get_entry(uint64_t address, uint64_t& root_address) noexcept {
    operation_lock lock{_num_in_progress_operations};
    root_address = 0;
    return get_entry(_root, address, root_address);
  }

  bool insert(uint64_t address, const value_type& v) {
    operation_lock lock{_num_in_progress_operations};
    int unused_leaves_delta = 0;
    return insert(_root, address, v, unused_leaves_delta);
  }

  bool erase(uint64_t address) {
    bool result = false;
    {
      operation_lock lock{_num_in_progress_operations};
      int unused_leaves_delta = 0;
      result = erase(_root, address, unused_leaves_delta);
    }
    if(result) {
      if(needs_garbage_collection()) {
        gc_lock lock{_num_in_progress_operations};
        if(needs_garbage_collection()) {
          garbage_collect();
        }
      }
    }
    return result;
  }

  ~allocation_map() {
    for(int i = 0; i < get_num_entries_in_level(root_level_idx); ++i) {
      auto* ptr = _root.children[i].load(std::memory_order::memory_order_acquire);
      if(ptr)
        release(*ptr);
    }
  }
    
private:

  static constexpr int num_levels = 16;
  static constexpr int root_level_idx = num_levels - 1;
  static constexpr int bitsizes[num_levels] = {4, 4, 4, 4, 4, 4, 4, 4,
                                               4, 4, 4, 4, 4, 4, 4, 4};
  static constexpr int empty_leaves_gc_trigger = 10000;

  static constexpr int get_num_entries_in_level(int level) {
    return (1ull << (bitsizes[level] + 1)) - 1;
  }

  static constexpr int get_bitoffset_in_level(int level) {
    int result = 0;
    for(int i = 0; i < level; ++i) {
      result += bitsizes[i];
    }
    return result;
  }

  static constexpr int get_index_in_level(uint64_t address, int level) {
    uint64_t bitmask = (1ull << (bitsizes[level] + 1)) - 1;
    return (address >> get_bitoffset_in_level(level)) & bitmask;
  }

  struct leaf_node {
    leaf_node() {
      for(int i = 0; i < get_num_entries_in_level(0); ++i) {
        entries[i].allocation_size = 0;
      }
    }

    value_type entries [get_num_entries_in_level(0)];
    std::atomic<int> num_entries;
  };

  template<int Level>
  struct intermediate_node {
  private:
      static constexpr auto make_child() {
      if constexpr (Level > 1) return 
        intermediate_node<Level - 1>{};
      else return leaf_node{};
    }
  public:
    intermediate_node()
    : children{}, num_empty_leaves{} {}

    using child_type = decltype(make_child());

    std::atomic<child_type*> children [get_num_entries_in_level(Level)];
    std::atomic<int> num_empty_leaves [get_num_entries_in_level(Level)];
  };

  template<class T>
  static T* alloc(int count) {
    return static_cast<T*>(__libc_malloc(sizeof(T) * count));
  }

  static void free(void* ptr) {
    __libc_free(ptr);
  }

  value_type *get_entry(leaf_node &current_node,
                        uint64_t address, uint64_t &root_address) noexcept {
    for (int local_address = get_index_in_level(address, 0); local_address >= 0;
         --local_address) {
      auto& element = current_node.entries[local_address];
      if(element.allocation_size > 0) {
        root_address |= static_cast<uint64_t>(local_address)
                        << get_bitoffset_in_level(0);
        if(address >= root_address) {
          return &element;
        } else {
          return nullptr;
        }
        
      }
    }
    return nullptr;
  }

  template <int Level>
  value_type *get_entry(intermediate_node<Level> &current_node,
                        uint64_t address, uint64_t &root_address) noexcept {
    for (int local_address = get_index_in_level(address, Level);
         local_address >= 0; --local_address) {
      
      auto *ptr = current_node.children[local_address].load(
          std::memory_order::memory_order_acquire);
      
      if(ptr) {
        root_address |= static_cast<uint64_t>(local_address)
                        << get_bitoffset_in_level(Level);
        return get_entry(*ptr, address, root_address);
      }
    }
    return nullptr;
  }

  bool insert(leaf_node &current_node, uint64_t address, const value_type &v,
              int &unused_leaves_delta) {
    unused_leaves_delta = 0;

    int local_address = get_index_in_level(address, 0);
    current_node.entries[local_address] = v;
    
    auto val = current_node.num_entries.fetch_add(
        1, std::memory_order::memory_order_acq_rel);
    if (val == 0) {
      _num_empty_leaves.fetch_add(-1, std::memory_order_acq_rel);
      unused_leaves_delta = -1;
    }
    return true;
  }

  template <int Level>
  bool insert(intermediate_node<Level> &current_node, uint64_t address,
              const value_type &v, int &unused_leaves_delta) {
    using child_t = typename intermediate_node<Level>::child_type;

    int local_address = get_index_in_level(address, Level);
    
    auto *ptr = current_node.children[local_address].load(
        std::memory_order::memory_order_acquire);
    
    if(!ptr) {
      child_t* new_child = alloc<child_t>(1);
      new (new_child) child_t{};

      if (!current_node.children[local_address].compare_exchange_strong(
              ptr, new_child, std::memory_order_acq_rel)) {
        new_child->~child_t();
        free(new_child);
      } else {
        ptr = new_child;
      }
    }

    int delta = 0;
    auto result = insert(*ptr, address, v, delta);
    if(delta != 0) {
      current_node.num_empty_leaves[local_address].fetch_add(
          delta, std::memory_order_acq_rel);
    }
    unused_leaves_delta = delta;
    return result;
  }

  bool erase(leaf_node& current_node, uint64_t address, int& unused_leaves_delta) {
    int local_address = get_index_in_level(address, 0);
    current_node.entries[local_address].allocation_size = 0;

    int count = current_node.num_entries.fetch_sub(
        1, std::memory_order::memory_order_acq_rel);
    
    unused_leaves_delta = 0;
    if (count == 1 /* has just become empty */) {
      _num_empty_leaves.fetch_add(1, std::memory_order::memory_order_acq_rel);
      unused_leaves_delta = 1;
    }
    return true;
  }

  template<int Level>
  bool erase(intermediate_node<Level>& current_node, uint64_t address, int& unused_leaves_delta) {
    int local_address = get_index_in_level(address, Level);
    auto *ptr = current_node.children[local_address].load(
        std::memory_order::memory_order_acquire);
    if(!ptr)
      return false;
    
    int delta = 0;
    bool result = erase(*ptr, address, delta);
    if(delta != 0) {
      current_node.num_empty_leaves[local_address].fetch_add(
          delta, std::memory_order_acq_rel);
    }
    unused_leaves_delta = delta;

    return result;
  }

  void release(leaf_node& current_node) {
    destroy(current_node);
  }

  template<int Level>
  void release(intermediate_node<Level>& current_node) {
    for(int i = 0; i < get_num_entries_in_level(Level); ++i){
      if (auto *ptr = current_node.children[i].load(
              std::memory_order::memory_order_acquire)) {
        release(*ptr);
        free(ptr);
      }
    }
    destroy(current_node);
  }

  bool needs_garbage_collection() const noexcept {
    return _num_empty_leaves.load(std::memory_order_acquire) >= empty_leaves_gc_trigger;
  }

  void destroy(leaf_node& node) {
    node.~leaf_node();
  }

  template<int Level>
  void destroy(intermediate_node<Level>& node) {
    node.~intermediate_node<Level>();
  }

  bool garbage_collect(leaf_node& current_node) {
    if(current_node.num_entries.load(std::memory_order::memory_order_acquire) == 0) {
      return true;
    }
    return false;
  }

  template<int Level>
  bool garbage_collect(intermediate_node<Level>& current_node) {
    int num_living_children = 0;
    for(int i = 0; i < get_num_entries_in_level(Level); ++i) {
      auto* child_ptr = current_node.children[i].load(std::memory_order::memory_order_acquire);

      if(child_ptr) {
        ++num_living_children;

        if(current_node.num_empty_leaves[i] > 0) {
          if(garbage_collect(*child_ptr)) {
            destroy(*child_ptr);
            free(child_ptr);
            current_node.children[i].store(nullptr, std::memory_order::memory_order_release);
            --num_living_children;
          }
          // After GC, num empty nodes is always 0
          current_node.num_empty_leaves[i] = 0; 
        }
      }
    }
    return num_living_children == 0;
  }

  void garbage_collect() {
    garbage_collect(_root);
  }

  struct gc_lock {
  public:
    gc_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = 0;
      while (!_op_counter.compare_exchange_strong(
          expected, -1, std::memory_order::memory_order_release,
          std::memory_order_relaxed)) {
        expected = 0;
      }
    }

    ~gc_lock() {
      _op_counter.store(0, std::memory_order::memory_order_release);
    }
  private:
    std::atomic<int>& _op_counter;
  };

  struct operation_lock {
  public:
    operation_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = std::max(0, _op_counter.load(std::memory_order_acquire));
      while (!_op_counter.compare_exchange_strong(
          expected, expected+1, std::memory_order::memory_order_release,
          std::memory_order_relaxed)) {
        if(expected < 0)
          expected = 0;
      }
    }

    ~operation_lock() {
      _op_counter.fetch_sub(1, std::memory_order::memory_order_acq_rel);
    }
  private:
   std::atomic<int>& _op_counter;
  };

  intermediate_node<root_level_idx> _root;
  
  std::atomic<int> _num_in_progress_operations;
  std::atomic<int> _num_empty_leaves;
};

}

#endif
