template<typename T>
void foo(T);

template <typename T>
void bar(T) {}

template<typename T>
void foo(T value) {
    bar(value);
}

int main() {
    return 0;
} 
