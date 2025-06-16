#include "hipSYCL/sycl/tracer_utils.hpp"
#include <iostream>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct submission_state_t {
  int num_submissions_start = 0;
  int num_submissions_end = 0;
};

void submission_start(void *submission_state) {
  ((submission_state_t *)submission_state)->num_submissions_start++;
  std::cout << "Hello World from the submission start function!" << std::endl;
}

void submission_end(void *submission_state) {
  ((submission_state_t *)submission_state)->num_submissions_end++;
  std::cout << "Hello World from the submission end function!" << std::endl;
}

void init_register() {
  submission_state_t *submission_state = new submission_state_t;
  init_submit_state(submission_state);
  init_submit_start(submission_start);
  init_submit_end(submission_end);
}

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */
