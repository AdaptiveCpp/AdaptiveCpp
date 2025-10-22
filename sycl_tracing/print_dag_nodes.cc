#include "hipSYCL/sycl/tracer_utils.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using timepoint_dur = std::chrono::duration<double, std::micro>;

template <typename T> struct TD;

struct queue_t;
struct dag;

struct event_node {
  using hashtype = decltype(std::hash<void *>{}(nullptr));
  hashtype id = 0;
  timepoint_dur time_created;
  std::shared_ptr<timepoint_dur> time_submitted = nullptr;
  std::shared_ptr<queue_t> parent_queue = nullptr;
  dag *parent_dag = nullptr;
  std::vector<std::shared_ptr<event_node>> dependecy_events{};
  bool complete = false;
  bool valid = false;

  friend bool operator==(const event_node &this_evt, const event_node &other) {
    return this_evt.id == other.id && this_evt.time_created == other.time_created;
  }

  friend bool operator!=(const event_node &this_evt, const event_node &other) {
    return !(this_evt == other);
  }

  void set_waited_for();
};

template <> struct std::hash<event_node> {
  std::size_t operator()(const event_node &evt) const noexcept {
    return std::hash<event_node::hashtype>{}(evt.id);
  }
};

struct queue_t {
  hashtype id;
  bool valid = false;
  timepoint time_created;
  std::shared_ptr<event_node> most_recent_event = nullptr;
  bool is_in_order;
  std::unordered_set<std::shared_ptr<event_node>> all_events;
  std::unordered_set<std::shared_ptr<event_node>> incomplete_events;

  friend bool operator==(const queue_t &this_queue, const queue_t &other_queue) {
    return this_queue.id == other_queue.id && this_queue.time_created == this_queue.time_created;
  }

  friend bool operator!=(const queue_t &this_queue, const queue_t &other_queue) {
    return !(this_queue == other_queue);
  }
};

template <> struct std::hash<queue_t> {
  std::size_t operator()(const queue_t &queue) const noexcept {
    return std::hash<event_node::hashtype>{}(queue.id);
  }
};

struct dag {
  std::unordered_set<std::shared_ptr<queue_t>>
      all_queues; // All queue that have ever existed in the program
  std::unordered_map<event_node::hashtype, std::shared_ptr<queue_t>> valid_queues;
  std::unordered_set<std::shared_ptr<event_node>> all_events;
  std::unordered_map<event_node::hashtype, std::shared_ptr<event_node>> valid_events;
  std::unordered_set<std::shared_ptr<event_node>> incompleted_events;
  std::vector<std::shared_ptr<event_node>> current_dependcies;
  timepoint time_created;

  std::ofstream outfile;

  dag(std::string filename)
      : outfile(filename), time_created(std::chrono::high_resolution_clock::now()) {}
};

std::ostream &operator<<(std::ostream &os, event_node event) {

  if (event.time_submitted.get()) {

    os << "{\"name\": \"Event" << event.id << event.time_created.count() << "\""
       << ", \"cat\": " << (event.complete ? "\"Complete\"" : "\"Incomplete\"")
       << ", \"ph\": \"B\","
       << "\"ts\":" << event.time_created.count() << ", \"pid\":" << event.parent_queue->id
       << ", \"tid\":" << 1 << "}," << std::endl;

    os << "{\"name\": \"Event" << event.id << event.time_created.count() << "\""
       << ", \"cat\": " << (event.complete ? "\"Complete\"" : "\"Incomplete\"")
       << ", \"ph\": \"E\","
       << "\"ts\":" << event.time_submitted->count() << ", \"pid\":" << event.parent_queue->id
       << ", \"tid\":" << 1 << "}," << std::endl;

    for (auto dep_event : event.dependecy_events) {

      auto first_time_point =
          (dep_event->time_created.count() + dep_event->time_submitted->count()) / 2;

      auto second_time_point = (event.time_created.count() + event.time_submitted->count()) / 2;

      os << "{\"name\": \"dependency" << event.id << event.time_created.count() << "_"
         << dep_event->id << dep_event->time_created.count() << "\","
         << "\"cat\":\"dependency\", \"id\" : 1, \"ph\":\"s\", \"ts\":" << first_time_point
         << ", \"pid\":" << dep_event->parent_queue->id << ", \"tid\": 1 " << "}," << std::endl;

      os << "{\"name\": \"dependency" << event.id << event.time_created.count() << "_"
         << dep_event->id << dep_event->time_created.count() << "\","
         << "\"cat\":\"dependency\", \"id\" : 1,  \"ph\":\"t\", \"ts\":" << second_time_point
         << ", \"pid\":" << event.parent_queue->id << ", \"tid\": 1 "
         << "}," << std::endl;
    }
  }

  return os;
};

void event_node::set_waited_for() {
  this->complete = true;

  auto dummy_shared_this = std::shared_ptr<event_node>{this, [](auto *) {}};
  auto event_iterator = parent_dag->incompleted_events.find(dummy_shared_this);
  if (event_iterator != parent_dag->incompleted_events.end()) {
    parent_dag->incompleted_events.erase(event_iterator);
  } else {
    std::cout << "Event not in the list of incompleted events of the dag node. (Implementation bug)"
              << std::endl;
  }

  if (parent_queue.get() != nullptr) {
    auto parent_queue_event_iterator = parent_queue->incomplete_events.find(dummy_shared_this);
    if (parent_queue_event_iterator != parent_queue->incomplete_events.end()) {
      parent_queue->incomplete_events.erase(parent_queue_event_iterator);
    } else {
      std::cout << "Event not in the list of incompleted events of the parent queue node. "
                   "(Implementation bug)"
                << std::endl;
    }
  }

  for (auto &event : dependecy_events) {
    if (!(event->complete)) {
      event->set_waited_for();
    }
  }
}

#ifdef __cplusplus
extern "C" {
#endif

void wait_function_queue(void *usr_state, event_node::hashtype queue_id) {
  dag *dagraph = (dag *)usr_state;

  std::cout << "While waiting the queue_id is: " << queue_id << std::endl;

  auto queue_it = dagraph->valid_queues.find(queue_id);
  if (queue_it == dagraph->valid_queues.end()) {
    std::cout << "This queue is not registered" << std::endl;
  } else {

    auto queue = queue_it->second;
    while (queue->incomplete_events.size() > 0) {
      auto event_it = queue->incomplete_events.begin();
      (*event_it)->set_waited_for();
    }
  }
}

void wait_function_event(void *usr_state, event_node::hashtype event_id) {
  dag *dagraph = (dag *)usr_state;
  auto event = dagraph->valid_events.find(event_id);

  if (event != dagraph->valid_events.end()) {
    event->second->set_waited_for();
  } else {
    std::cout << "Event is already complete" << std::endl;
  }
}

void queue_impl_construction(void *usr_state, event_node::hashtype queue_id, bool is_in_order) {

  std::cout << "At construction the queue id is: " << queue_id << std::endl;

  auto *dag_ptr = (dag *)usr_state;
  auto shared_queue =
      std::make_shared<queue_t>(queue_t{.id = queue_id,
                                        .valid = true,
                                        .time_created = std::chrono::high_resolution_clock::now(),
                                        .is_in_order = is_in_order});

  dag_ptr->all_queues.insert(shared_queue);
  dag_ptr->valid_queues.insert({queue_id, shared_queue});
}

void queue_impl_destruction(void *usr_state, event_node::hashtype queue_id) {
  auto *dag_ptr = (dag *)usr_state;
  auto key_queue_pair = dag_ptr->valid_queues.find(queue_id);

  if (key_queue_pair != dag_ptr->valid_queues.end()) {
    dag_ptr->valid_queues.erase(key_queue_pair);
    key_queue_pair->second->valid = false;
  } else {
    std::cout << "Destructor called on invalid queue: " << std::endl;
  }
}

void dag_node_constructor(void *usr_state, event_node::hashtype event_hash) {
  auto *dag_ptr = (dag *)usr_state;
  auto shared_event = std::make_shared<event_node>(event_node{
      .id = event_hash,
      .time_created = std::chrono::high_resolution_clock::now() - (dag_ptr->time_created)});

  shared_event->parent_dag = dag_ptr;
  dag_ptr->all_events.insert(shared_event);

  {
    auto previous_holder_of_hash = dag_ptr->valid_events.find(event_hash);

    if (previous_holder_of_hash != dag_ptr->valid_events.end()) {
      std::cout << "New event created before previous event with same hash has been invalidated, "
                   "looks like a bug"
                   "Discarding the previous one"
                << std::endl;
    }
  }

  dag_ptr->valid_events[event_hash] = shared_event;
}

void dag_node_destructor(void *usr_state, event_node::hashtype event_hash) {
  auto *dag_ptr = (dag *)usr_state;

  {
    auto invalidated_event = dag_ptr->valid_events.find(event_hash);
    if (invalidated_event != dag_ptr->valid_events.end()) {
      dag_ptr->valid_events.erase(invalidated_event);
    } else {
      std::cout << "Deleting invalidated event not part of the list" << std::endl;
    }
  }
}

void submit_end_function(void *usr_state, event_node::hashtype event_hash,
                         event_node::hashtype queue_id) {
  auto dag_ptr = (dag *)usr_state;
  auto queue = dag_ptr->valid_queues[queue_id];
  auto event = dag_ptr->valid_events[event_hash];
  event->time_submitted = std::make_shared<timepoint_dur>(
      std::chrono::high_resolution_clock::now() - dag_ptr->time_created);

  // Setting up criss cross dependencies
  dag_ptr->incompleted_events.insert(event);
  queue->incomplete_events.insert(event);
  event->parent_queue = dag_ptr->valid_queues[queue_id];

  if (queue->is_in_order && (queue->most_recent_event)) {
    // std::cout << "This queue is an in order queue and has a most recent event" << std::endl;
    dag_ptr->current_dependcies.push_back(queue->most_recent_event);
  }

  queue->most_recent_event = event;
  event->dependecy_events = std::move(dag_ptr->current_dependcies);
  dag_ptr->current_dependcies.clear();

  // dag_ptr->outfile <<
}

void depends_on_end_function(void *usr_state, event_node::hashtype event_hash) {
  auto dag_ptr = (dag *)usr_state;
  auto event = dag_ptr->valid_events[event_hash];
  dag_ptr->current_dependcies.push_back(event);
}

void finalize(void *usr_state) {
  dag *dag_ptr = (dag *)usr_state;
  for (auto event : dag_ptr->all_events) {
    dag_ptr->outfile << *event;
  }

  dag_ptr->outfile << std::endl;
  dag_ptr->outfile << "]} " << std::endl;

  dag_ptr->outfile.close();
}

void init_register() {
  void *usr_state = new dag("Event_trace.json");
  ((dag *)usr_state)->outfile << "{" << std::endl;
  ((dag *)usr_state)->outfile << "  \"traceEvents\":[" << std::endl;
  //   init_submit_start(queue_submit_start_function);
  init_queue_impl_constructor(queue_impl_construction);
  init_queue_impl_destructor(queue_impl_destruction);
  init_dag_node_constructor(dag_node_constructor);
  init_dag_node_destructor(dag_node_destructor);
  init_submit_end(submit_end_function);
  init_wait_queue_end(wait_function_queue);
  init_wait_event_end(wait_function_event);
  //   init_depends_on_start(depends_on_start_function);
  init_depends_on_end(depends_on_end_function);
  init_states(usr_state);
  init_finalize(finalize);
}

#ifdef __cplusplus
}
#endif
