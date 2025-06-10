
#include <stdio.h>

enum start_end {
  START, // 0
  END    // 1
};

enum tracer_type {
  SUBMIT,
  SUBMIT_SECONDARY,
  PARALLEL_FOR,
  PARALLEL_FOR_WORK_GROUP,
  SINGLE_TASK,
  MEMCPY,
  MEMSET
};

void tracer(enum tracer_type trace, enum start_end se_val) {
  printf("Tracer function called\n");
  printf("Tracer type: %d\n", trace);
  printf("Start/End: %d\n", se_val);
}
