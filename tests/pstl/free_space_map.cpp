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

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <cstdint>
#include <random>
#include <unordered_set>
#include <thread>
#include <hipSYCL/std/stdpar/detail/allocation_map.hpp>


#include "pstl_test_suite.hpp"

BOOST_AUTO_TEST_SUITE(pstl_free_space_map)

using amap_t = hipsycl::stdpar::allocation_map<>;
using fmap_t = hipsycl::stdpar::free_space_map;

uint64_t next_pow2(uint64_t x) {
  for(int i = 0; i < 64; ++i) {
    if((1ull << i) >= x)
      return (1ull << i);
  }
  return 0;
}

BOOST_AUTO_TEST_CASE(insert) {
  std::size_t alloc_space_size = 1ull << 20;
  fmap_t fmap{alloc_space_size};
  amap_t amap;

  std::mt19937 gen(123);
  std::uniform_int_distribution<uint64_t> dist{1, 1ull << 12};
  std::size_t alloced_size = 0;

  for(int i = 0; i < 2048 && alloced_size < 0.9 * alloc_space_size; ++i) {

    uint64_t addr;
    std::size_t size = dist(gen);
    bool res = fmap.claim(size, addr);

    BOOST_CHECK(res);

    if(res) {
      uint64_t root_addr = 0;
      BOOST_CHECK(!amap.get_entry(addr, root_addr));
      BOOST_CHECK(!amap.get_entry(addr+size-1, root_addr));
      BOOST_CHECK(addr+size < alloc_space_size);

      amap_t::value_type v;
      v.allocation_size = size;
      amap.insert(addr, v);

      alloced_size += next_pow2(size);
    }
  }
}

BOOST_AUTO_TEST_CASE(insert_erase_reclaim) {
  std::size_t alloc_space_size = 1ull << 20;
  fmap_t fmap{alloc_space_size};

  std::mt19937 gen(123);
  std::uniform_int_distribution<uint64_t> dist{1, 1ull << 12};
  std::size_t alloced_size = 0;

  for(int i = 0; i < 2048 && alloced_size < 0.9 * alloc_space_size; ++i) {
    uint64_t addr1, addr2;
    std::size_t size1 = dist(gen);
    std::size_t size2 = dist(gen);
    bool res1 = fmap.claim(size1, addr1);
    BOOST_CHECK(res1);
    bool res2 = fmap.claim(size2, addr2);
    BOOST_CHECK(res2);

    if(res1)
      alloced_size += next_pow2(size1);
    if(res2)
      alloced_size += next_pow2(size2);

    if(res2) {
      BOOST_CHECK(fmap.release(addr2, size2));
      uint64_t addr3 = 0;
      BOOST_CHECK(fmap.claim(size2, addr3));
      BOOST_CHECK(addr3 == addr2);
    }
  }
}

BOOST_AUTO_TEST_CASE(hybrid_insert_erase) {
  std::size_t alloc_space_size = 1ull << 20;
  fmap_t fmap{alloc_space_size};
  amap_t amap;

  std::mt19937 gen(123);
  std::uniform_int_distribution<uint64_t> dist{1, 1ull << 12};

  std::vector<std::pair<uint64_t,uint64_t>> allocs;
  std::size_t alloced_size = 0;

  int erase_delay = 800;

  for(int i = 0; i < 2048 && alloced_size < 0.9 * alloc_space_size; ++i) {
    uint64_t addr1, addr2;
    std::size_t size1 = dist(gen);
    std::size_t size2 = dist(gen);
    bool res1 = fmap.claim(size1, addr1);
    BOOST_CHECK(res1);
    bool res2 = fmap.claim(size2, addr2);
    BOOST_CHECK(res2);

    uint64_t root_addr = 0;
    amap_t::value_type v;
    if(res1) {
      BOOST_CHECK(!amap.get_entry(addr1, root_addr));
      BOOST_CHECK(!amap.get_entry(addr1+size1-1, root_addr));
      v.allocation_size = size1;
      amap.insert(addr1, v);
      allocs.push_back(std::make_pair(addr1, size1));
      alloced_size += next_pow2(size1);
    }
    if(res2) {
      BOOST_CHECK(!amap.get_entry(addr2, root_addr));
      BOOST_CHECK(!amap.get_entry(addr2+size2-1, root_addr));
      v.allocation_size = size2;
      amap.insert(addr2, v);
      allocs.push_back(std::make_pair(addr2, size2));
      alloced_size += next_pow2(size2);
    }

    int erase_pos = i - erase_delay;
    if(erase_pos >= 0) {
      auto alloc = allocs[erase_pos];
      BOOST_CHECK(fmap.release(alloc.first, alloc.second));
      amap.erase(alloc.first);
      alloced_size -= next_pow2(alloc.second);
    }
  }
}



BOOST_AUTO_TEST_SUITE_END()
