class foo {
    template<typename T>
    void bar(T value);
    
    template<typename T>
    int func_with_return_value();
};

template <typename T>
void baz(T) {}

template <typename T>
void foo::bar(T value) {
    baz(value);
}

template <typename T>
int foo::func_with_return_value() {
    baz(T{});
    return T{};
}


int main() {
    return 0;
}
