#include <iostream>

extern void (*tracer_func)();

// void tracer_func() { std::cout << "tracer_func called" << std::endl; }

int main() {
  std::cout << "Hello, World!" << std::endl;
  tracer_func();
  return 0;
}
