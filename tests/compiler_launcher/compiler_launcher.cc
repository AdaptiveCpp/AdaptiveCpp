#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
  std::cout << "Hello from compiler launcher\n";

  std::string command;
  for (int i=1; i < argc; ++i) {
    command += argv[i];
    command += " ";
  }

  std::system(command.c_str());
}
