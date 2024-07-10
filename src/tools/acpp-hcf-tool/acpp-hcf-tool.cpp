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
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "hipSYCL/common/hcf_container.hpp"

void help() {
  std::cout <<
  "Usage: acpp-hcf-tool <hcf-file> <-x|-r <file>|-p> root [subnode] [subsubnode] ...\n" <<
  "  -x: Extract binary attachment and print to stdout\n" <<
  "  -r <file>: Replace binary attachment with file content and print to stdout\n" << 
  "  -p: Print node content" << std::endl;
}

enum class mode {
  print_node_content,
  print_attachment,
  replace_attachment
};

bool read_file(const std::string& filename, std::string& out) {
  std::ifstream file{filename, std::ios::binary|std::ios::ate};
  if(!file.is_open())
    return false;

  auto size = file.tellg();

  if (size == 0) {
      out = std::string{};
      return true;
  }

  std::string result(size, '\0');

  file.seekg(0, std::ios::beg);
  file.read(result.data(), size);

  out = result;

  return true;
}

void print_node(const hipsycl::common::hcf_container::node* n, int level = 0) {

  std::string prefix;
  for(int i=0; i < level; ++i)
    prefix += ' ';
  
  for(const auto& kv_pair : n->key_value_pairs) {
    std::cout << prefix << kv_pair.first << " = " << kv_pair.second << std::endl;
  }

  for(const auto& subnode : n->subnodes) {
    std::cout << prefix << subnode.node_id << ":" << std::endl;
    print_node(&subnode, level+1);
  }
}

template <class AttachmentHandler>
bool copy_node(const hipsycl::common::hcf_container::node *source,
               hipsycl::common::hcf_container::node *target,
               AttachmentHandler &&h) {

  for (const auto &kv_pair : source->key_value_pairs) {
    target->set(kv_pair.first, kv_pair.second);
  }

  for (auto &subnode : source->subnodes) {
    if (!subnode.is_binary_content()) {
      auto* new_subnode = target->add_subnode(subnode.node_id);

      if(!new_subnode)
        return false;

      return copy_node(&subnode, new_subnode, h);
    } else {
      if(!h(source, target))
        return false;
    }
  }
  return true;
};

int main(int argc, char** argv) {
  std::vector<std::string> args;
  for(int i = 1; i < argc; ++i) {
    args.push_back(std::string{argv[i]});
  }

  if(args.size() < 3) {
    help();
    return -1;
  }

  std::string filename = args[0];

  std::string hcf_content;
  if(!read_file(filename, hcf_content)) {
    std::cout << "Could not read file: " << filename << std::endl;
    return -1;
  }
  hipsycl::common::hcf_container hcf{hcf_content};

  mode m = mode::print_node_content;
  std::string replacement_filename;
  int target_node_start = 2;

  if(args[1] == "-x") {
    m = mode::print_attachment;
  }
  else if(args[1] == "-r") {
    m = mode::replace_attachment;
    replacement_filename = args[2];
    ++target_node_start;
  } else if(args[1] == "-p") {
    m = mode::print_node_content;
  } else {
    std::cout << "Unknown mode of operation: " << args[1] << std::endl;
    return -1;
  }

  std::vector<std::string> node_descriptor;
  for(int i = target_node_start; i < args.size(); ++i) {
    node_descriptor.push_back(args[i]);
  }

  if(node_descriptor.empty()) {
    std::cout << "No target node was provided." << std::endl;
    help();
    return -1;
  }

  if(!hcf.root_node()) {
    std::cout << "Invalid root node. Is HCF corrupted?" << std::endl;
    return -1;
  }

  hipsycl::common::hcf_container::node* current = nullptr;

  for(int i = 0; i < node_descriptor.size(); ++i) {
    if(i == 0) {
      if(node_descriptor[i] == "root") {
        current = hcf.root_node();
      } else {
        std::cout << "Top level node name of target node is not 'root'; all node "
                 "descriptors must start at the root node"
              << std::endl;
        return -1;
      }
    } else {
      current = current->get_subnode(node_descriptor[i]);
    }

    if(!current) {
      std::cout << "Could not access specified node/subnode: " << node_descriptor[i] << std::endl;
      return -1; 
    }
  }

  if(m == mode::print_node_content) {
    print_node(current);
  } else if(m == mode::print_attachment) {
    if(!current->has_binary_data_attached()) {
      std::cout << "Specified node does not have binary data attached." << std::endl;
      return -1;
    }
    std::string attachment;
    if(!hcf.get_binary_attachment(current, attachment)) {
      std::cout << "Could not extract binary attachment." << std::endl;
      return -1;
    } else {
      std::cout << attachment;
    }
  } else if(m == mode::replace_attachment) {

    std::string content;
    if(!read_file(replacement_filename, content)) {
      std::cout << "Could not read file: " << replacement_filename << std::endl;
      return -1;
    }

    if(!current->has_binary_data_attached()) {
      hcf.attach_binary_content(current, content);
      std::cout << hcf.serialize();
    } else {
      hipsycl::common::hcf_container new_container;

      auto binary_attachment_handler =
          [&](const hipsycl::common::hcf_container::node *source,
              hipsycl::common::hcf_container::node *target) {
            assert(source->has_binary_data_attached());

            if(current == source) {
              new_container.attach_binary_content(target, content);
            } else {
              std::string attachment;
              if(!hcf.get_binary_attachment(source, attachment))
                return false;
              new_container.attach_binary_content(target, attachment);
            }

            return true;
          };

      if (!copy_node(hcf.root_node(), new_container.root_node(),
                     binary_attachment_handler)) {
        std::cout << "Constructing new HCF container failed" << std::endl;
        return -1;
      }

      std::cout << new_container.serialize();
    }
  }

  return 0;
}
