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

#ifndef HIPSYCL_OBJECT_POOL_HPP
#define HIPSYCL_OBJECT_POOL_HPP

#include <vector>
#include <atomic>

namespace hipsycl {
namespace common {

/// Provides an object pool of a given size to reduce memory allocations.
///
/// The pool size is provided when constructing the pool. The idea is to store
/// - a vector of objects
/// - a vector of indices in the object vector that are currently unoccupied ("free slots").
/// Both vectors are constructed of the size of the pool in the object pool constructor,
/// and not resized afterwards.
/// 
/// An atomic pointer points to the last element in the free slots vector that is used.
/// This pointer will be used to push or pop a slot to/from the free slots vector.
///
/// When an allocation is performed but the object pool is already full,
/// the object pool will construct another, larger pool that is referenced by the first object pool.
/// Thus, all object pools will form a linked list.
/// While this guarantees that the object pool can adapt to some extent to runtime
/// requirements, it is still desirable to choose a pool size that is large enough
/// for all allocations to fit into in the first place. In this case, only a single 
/// pool will be created, and the allocation process will not require iterating
/// through the object pool linked list which can be detrimental for performance.
///
/// Another important property of this approach is that pointers to allocations
/// will be stable and remain correct even if the number of allocations exceeds the
/// original pool size.
///
/// Because multiple object pools may be involved under the hood, the allocation
/// function returns not only the pointer to the allocation, but also the pool
/// from which this allocation originates. The pool pointer is needed when freeing
/// an object, because object pools can currently only free objects from their own storage!
template<class T>
class object_pool {
public:
  using object_id = std::size_t;

  object_pool(std::size_t pool_size)
      : _objects(pool_size), _free_slots(pool_size), _next_pool{nullptr} {
    for(std::size_t i = 0; i < pool_size; ++i)
      _free_slots[i] = i;
    
    _last_free_slot_ptr = _free_slots.data() + _free_slots.size() - 1;
  }

  object_pool(const object_pool&) = delete;
  object_pool(object_pool &&) = delete;
  object_pool& operator=(const object_pool&) = delete;
  object_pool& operator=(object_pool&&) = delete;

  ~object_pool() {
    if(_next_pool) {
      delete _next_pool;
    }
  }

  std::pair<object_pool<T>*, T*> alloc() {
    object_id obj = 0;
    if(!atomic_free_slot_pop(obj)) {
      return get_or_create_next_pool()->alloc();
    }

    return std::make_pair(this, &(_objects[obj]));
  }

  void local_free(T* obj_ptr) noexcept {
    if (obj_ptr < _objects.data() ||
        obj_ptr >= (_objects.data() + _objects.size()))
      return;
    
    object_id obj = obj_ptr - _objects.data();

    // This is a replacement for calling the destructor.
    // This might better be solved using placement new/delete
    //
    // Note: The weak_ptr implementation assumes that the memory
    // content is reset in this way!
    _objects[obj] = T{};
    atomic_free_slot_push(obj);
  }
  


private:
  bool atomic_free_slot_pop(object_id& out) noexcept {
    while(true) {
      // Wait until a non-null value is in _last_free_slot_ptr
      object_id *free_slot_ptr =
          _last_free_slot_ptr.exchange(nullptr);
      
      if(free_slot_ptr) {
        if(free_slot_ptr < _free_slots.data()) {
          _last_free_slot_ptr = free_slot_ptr;
          return false;
        } else {
          out = *free_slot_ptr;
          _last_free_slot_ptr = free_slot_ptr - 1;
          return true;
        }
      }
    }

    return false;
  }

  void atomic_free_slot_push(object_id obj) noexcept {
    while(true) {
      // Wait until a non-null value is in _last_free_slot_ptr
      object_id *free_slot_ptr =
          _last_free_slot_ptr.exchange(nullptr);
      
      if(free_slot_ptr) {
        ++free_slot_ptr;
        *free_slot_ptr = obj;
        _last_free_slot_ptr = free_slot_ptr;

        return;
      }
    }
  }

  object_pool<T>* get_or_create_next_pool() {
    // We do not delete pools once they have been created,
    // so if we have a next pool, we can always return it.
    if(_next_pool)
      return _next_pool;

    object_pool<T>* next_pool = new object_pool<T>{2 * _objects.size()};
    object_pool<T>* expected = nullptr;
    if(_next_pool.compare_exchange_strong(expected, next_pool)) {
      return next_pool;
    } else {
      delete next_pool;
      return expected;
    }
  }

  std::vector<T> _objects;
  std::vector<object_id> _free_slots;
  std::atomic<object_id*> _last_free_slot_ptr;
  std::atomic<object_pool<T>*> _next_pool;
};

template<class T>
class refcounted_object_pool {
  // TODO: Consider aligning to cache line size (or maybe
  // already the storage in object_pool itself)
  struct payload {
    payload() noexcept
    : _ref_count{0}, unique_id{0}, pool{nullptr} {
    }

    payload(std::size_t uid, object_pool<T>* obj_pool) noexcept
    :  _ref_count{0}, unique_id{uid}, pool{obj_pool} {}

    bool retain_lock() noexcept {
      unsigned count = _ref_count;
      do
	    {
	      if (count == 0)
	        return false;
	    } while (!_ref_count.compare_exchange_strong(count, count + 1));

      return true;
    }

    void retain() noexcept {
      ++_ref_count;
    }

    void release_and_dispose() noexcept {
      if(--_ref_count == 0) {
        pool->local_free(this);
      }
    }

    unsigned get_ref_count() noexcept {
      return _ref_count;
    }

    // Put data first to avoid having to add an offset
    // into the control block when accessing the data.
    T data;
    std::size_t unique_id;
    object_pool<T>* pool;
  private:
    std::atomic<unsigned> _ref_count;
  };

public:
  refcounted_object_pool(std::size_t pool_size)
  : _pool{pool_size} {}

  
  class shared_ptr {
  public:
    shared_ptr() noexcept
    : _data{nullptr} {}

    ~shared_ptr() noexcept {
      release();
    }

    shared_ptr(const shared_ptr& other) noexcept
    : shared_ptr{other._data} {
      retain();
    }

    // Construct shared_ptr on top of *existing* payload.
    // This is mainly used for the implementation of weak_ptr::lock().
    shared_ptr(payload* p) noexcept
    : _data{nullptr} {
      if(p && p->retain_lock())
        _data = p;
    }

    shared_ptr(payload* p, std::size_t uid, object_pool<payload*> pool) noexcept
    : _data{p} {
      *p = payload{uid, pool};
    }


    shared_ptr& operator=(const shared_ptr& other) noexcept {
      if(&other == this || other == *this)
        return *this;

      release();
      _data = other._data;
      retain();
      
      return *this;
    }

    shared_ptr& operator=(shared_ptr&& ptr) noexcept {
      shared_ptr{std::move(ptr)}.swap(*this);
	    return *this;
    }

    payload* get_payload() const noexcept {
      return _data;
    }

    T& operator*() noexcept {
      return _data->data;
    }

    const T& operator*() const noexcept {
      return _data->data;
    }

    T& operator->() noexcept {
      return _data->data;
    }

    const T& operator->() const noexcept {
      return _data->data;
    }

    operator bool() const noexcept {
      return _data;
    }

    void swap(shared_ptr& a) noexcept{
      std::swap(a._data, this->_data);
    }

    friend bool operator==(const shared_ptr& a, const shared_ptr& b) noexcept {
      return a._data == b._data;
    }

    friend bool operator!=(const shared_ptr& a, const shared_ptr& b) noexcept {
      return !(a == b);
    }
  private:
    void retain() noexcept {
      if(!_data)
        return;
      
      _data->retain();
    }

    void release() noexcept{
      if(!_data)
        return;
      
      _data->release_and_dispose();
    }

    mutable payload* _data;
  };

  class weak_ptr {
  public:
    weak_ptr() noexcept
    : _data{nullptr}, _id{0} {}

    weak_ptr(const shared_ptr& ptr) noexcept
    : _data{ptr.get_payload()}, _id{0} {
      if(_data) {
        _id = _data->unique_id;
      }
    }

    bool expired() const noexcept {
      if(!_data)
        return true;
      // This works because the object pool's local_free()
      // resets the memory content to a default-constructed object.
      return _data->unique_id != _id;
    }

    shared_ptr lock() const noexcept {
      
      if(!_data)
        return shared_ptr{};

      // This part here is a bit complicated:
      // In an atomic operation, we need to increment
      // the ref count if (and only if!) the unique id
      // matches. This is accomplished in two steps:
      // 1. Construct new shared_ptr from payload. It is important
      // to actually construct a shared_ptr such that freeing
      // is handled correctly in case the pointer has become invalid
      // and we are now looking at a different object.
      
      shared_ptr ptr{_data};
      
      // 2. Check if unique_id matches

      // This prevents us looking into the unique_id while
      // the shared_ptr destructor might be running and a free
      // might have partially completed.
      // (if ref_count == 1, there are no other users anymore except
      // for the shared_ptr we have just constructed)
      if(ptr.get_payload() && ptr.get_payload()->get_ref_count() > 1) {
        // TODO: Look into potential double free if free is already running
        if(ptr.get_payload()->unique_id == _id)
          return ptr;
      }
      
      return shared_ptr{};
    }
    
    void swap(weak_ptr& ptr) noexcept {
      std::swap(this->_data, ptr._data);
      std::swap(this->_id, ptr._id);
    }

    friend bool operator==(const weak_ptr& a, const weak_ptr& b) noexcept {
      return a._data == b._data && a._id == b._id;
    }

    friend bool operator!=(const shared_ptr& a, const shared_ptr& b) noexcept {
      return !(a == b);
    }
  private:
    mutable payload* _data;
    std::size_t _id;
  };

  shared_ptr make_shared(const T& data) {
    std::pair<object_pool<payload>*, payload*> allocation = _pool.alloc();
    shared_ptr ptr{allocation.second, generate_ptr_uid(), allocation.first};
    *ptr = data;
  }

private:
  std::size_t generate_ptr_uid() const {
    static std::atomic<std::size_t> counter;
    return ++counter;
  }

  object_pool<payload> _pool;
};

}
}

#endif
