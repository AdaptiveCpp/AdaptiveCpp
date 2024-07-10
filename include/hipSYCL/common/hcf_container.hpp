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
#ifndef HIPSYCL_HCF_CONTAINER_HPP
#define HIPSYCL_HCF_CONTAINER_HPP

#include "debug.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <algorithm>
#include <exception>
#include <cassert>
#include <locale>

namespace hipsycl {
namespace common {

class hcf_container {
public:
  struct node {
    std::vector<std::pair<std::string, std::string>> key_value_pairs;
    std::vector<node> subnodes;
    std::string node_id;

    const node* get_subnode(const std::string& name) const {
      for(int i = 0; i < subnodes.size(); ++i) {
        if(subnodes[i].node_id == name)
          return &(subnodes[i]);
      }
      return nullptr;
    }

    node* get_subnode(const std::string& name) {
      for(int i = 0; i < subnodes.size(); ++i) {
        if(subnodes[i].node_id == name)
          return &(subnodes[i]);
      }
      return nullptr;
    }

    const std::string* get_value(const std::string& key) const {
      for(int i = 0; i < key_value_pairs.size(); ++i) {
        if(key_value_pairs[i].first == key) {
          return &(key_value_pairs[i].second);
        }
      }
      return nullptr;
    }

    bool has_key(const std::string& key) const {
      return get_value(key) != nullptr;
    }

    bool has_subnode(const std::string& name) const {
      return get_subnode(name) != nullptr;
    }

    std::vector<std::string> get_subnodes() const {
      std::vector<std::string> result;
      for(const auto& s : subnodes) {
        result.push_back(s.node_id);
      }
      return result;
    }

    bool has_binary_data_attached() const {
      return has_subnode("__binary");
    }

    bool is_binary_content() const {
      return node_id == "__binary";
    }

    node* add_subnode(const std::string& unique_name) {
      for(int i = 0; i < subnodes.size(); ++i) {
        if(subnodes[i].node_id == unique_name) {
          HIPSYCL_DEBUG_ERROR << "hcf: Subnode already exists with name "
                              << unique_name << "\n";
          return nullptr;
        }
      }

      node new_node;
      new_node.node_id = unique_name;
      subnodes.push_back(new_node);
      return &subnodes.back();
    }

    void set(const std::string& key, const std::string& value) {
      key_value_pairs.push_back(std::make_pair(key, value));
    }

    // Note: This is a convenience feature. It just creates additional subnodes for list entries.
    void set_as_list(const std::string& key, const std::vector<std::string>& list_entries) {
      auto* N = add_subnode(key);
      if(N) {
        for(const auto& e : list_entries) {
          N->add_subnode(e);
        }
      }
    }

    // Returns vector of list entries if key is present, or empty vector
    // otherwise.
    // Note: This is a convenience feature. It just reads subnodes for list entries.
    std::vector<std::string> get_as_list(const std::string& key) const {
      if(!has_subnode(key))
        return {};
      return get_subnode(key)->get_subnodes();
    }
  };

  hcf_container() {
    _root_node.node_id = "root";
  }

  hcf_container(const std::string& container) {
    std::string appendix_id {_binary_appendix_id};

    std::size_t appendix_begin = container.find(appendix_id);
    if(appendix_begin != std::string::npos) {
      _binary_appendix = container.substr(appendix_begin + appendix_id.length());
    }

    std::string parseable_data = container;
    parseable_data.erase(appendix_begin);

    parse(parseable_data);
  }

  const node* root_node() const {
    return &_root_node;
  }

  node* root_node() {
    return &_root_node;
  }

  bool get_binary_attachment(const node* n, std::string& out) const {
    std::size_t start = 0;
    std::size_t size = 0;

    if(!n)
      return false;

    const node* descriptor_node = nullptr;

    if(n->is_binary_content())
      descriptor_node = n;
    else if(n->has_binary_data_attached()) {
      descriptor_node = n->get_subnode("__binary");
    } else {
      HIPSYCL_DEBUG_ERROR << "hcf: Node " << n->node_id
                          << " is not a binary content node, nor does it carry "
                             "a binary attachment\n";
      return false;
    }
    assert(descriptor_node);

    const std::string* start_entry = descriptor_node->get_value("start");
    const std::string* size_entry = descriptor_node->get_value("size");

    if(!start_entry) {
      HIPSYCL_DEBUG_ERROR << "hcf: Node does not contain binary content start\n";
      return false;
    }
    if(!size_entry) {
      HIPSYCL_DEBUG_ERROR << "hcf: Node does not contain binary content size\n";
      return false;
    }

    start = std::stoull(*start_entry);
    size = std::stoull(*size_entry);

    if(start + size > _binary_appendix.size()) {
      HIPSYCL_DEBUG_ERROR << "hcf: Binary content address is out-of-bounds\n";
      return false;
    }

    out = _binary_appendix.substr(start, size);

    return true;
  }

  bool attach_binary_content(node* n, const std::string& binary_content) {
    
    node* binary_node = n->add_subnode(_binary_marker);
    if(!binary_node)
      return false;

    std::size_t start = _binary_appendix.size();
    std::size_t length = binary_content.size();

    _binary_appendix += binary_content;

    binary_node->set("start", std::to_string(start));
    binary_node->set("size", std::to_string(length));

    return true;
  }

  std::string serialize() const {
    std::stringstream sstr;
    serialize_node(_root_node, sstr);
    sstr << _binary_appendix_id;

    return sstr.str() + _binary_appendix;
  }
private:

  void serialize_node(const node& n, std::ostream& out) const {
    for(const auto& p : n.key_value_pairs){
      out << p.first << "=" << p.second << "\n";
    }
    for(const auto& s : n.subnodes) {
      out << _node_start_id << s.node_id << "\n";
      serialize_node(s, out);
      out << _node_end_id  << s.node_id << "\n";
    }
  }

  void trim_left(std::string& str) const
  {
    auto it2 = std::find_if(str.begin(), str.end(), [](char ch) {
      return !std::isspace<char>(ch, std::locale::classic());
    });
    str.erase(str.begin(), it2);
  }

  void trim_right(std::string& str) const
  {
    auto it1 = std::find_if(str.rbegin(), str.rend(), [](char ch) {
      return !std::isspace<char>(ch, std::locale::classic());
    });
    str.erase(it1.base(), str.end());
  }

  bool parse_node_start(const std::string& start_line, std::string& node_id) const {
    std::string processed_data = start_line;
    std::string node_start_id = std::string{_node_start_id};

    trim_left(processed_data);
    
    if(processed_data.find(node_start_id) != 0) {
      HIPSYCL_DEBUG_ERROR << "hcf: Invalid node start: " << processed_data
                          << "\n";
      return false;
    }


    processed_data.erase(0, node_start_id.length());
    trim_right(processed_data);

    node_id = processed_data;

    return true;
  }

  bool parse_node_interior(const std::vector<std::string>& lines, 
                           std::size_t node_start_line,
                           std::size_t node_end_line,
                           node& current_node) const {

    if(node_start_line == node_end_line)
      return true;

    for(int i = node_start_line; i <  node_end_line; ++i) {
      assert(i < lines.size());
      const std::string& current = lines[i];

      if(current.find(_node_start_id) == 0) {
        node new_node;

        if(!parse_node_start(current, new_node.node_id)) {
          return false;
        }

        std::size_t num_node_lines = std::string::npos;
        std::string node_end_marker = _node_end_id + new_node.node_id;
        
        for(int j = i + 1; j < node_end_line; ++j) {
          if(lines[j] == node_end_marker) {
            num_node_lines = j-i;
            break;
          }
        }
        if(num_node_lines == std::string::npos) {
          HIPSYCL_DEBUG_ERROR
              << "hcf: Syntax error: Did not find expected node end marker: "
              << node_end_marker << "\n";
          return false;
        }
        if(!parse_node_interior(lines, i+1, i+num_node_lines, new_node))
          return false;

        current_node.subnodes.push_back(new_node);
        i += num_node_lines;
      } else if(current.find("=") != std::string::npos) {
        std::size_t pos = current.find("=");
        std::string key = current.substr(0, pos);
        std::string value = current.substr(pos+1);
        current_node.key_value_pairs.push_back(std::make_pair(key, value));
      } else if(current.find(_node_end_id) == 0) {
        HIPSYCL_DEBUG_ERROR << "hcf: Syntax error: Unexpected node end: " << current
                          << "\n";
        return false;
      } else {
        HIPSYCL_DEBUG_ERROR << "hcf: Syntax error: Invalid line: " << current
                          << "\n";
        return false;
      }

    }

    return true;
  }

  bool parse(const std::string& data) {
    std::stringstream ss(data);
    std::string line;

    std::vector<std::string> lines;
    
    while(std::getline(ss,line)){
      trim_left(line);
      trim_right(line);

      if(!line.empty())
        lines.push_back(line);
    }

    _root_node.node_id = "root";
    return parse_node_interior(lines, 0, lines.size(), _root_node);
  }

  static constexpr char _binary_appendix_id [] = "__acpp_hcf_binary_appendix";
  static constexpr char _node_start_id [] = "{.";
  static constexpr char _node_end_id [] = "}.";
  static constexpr char _binary_marker [] = "__binary";

  node _root_node;
  std::string _binary_appendix;
};

}
}

#endif
