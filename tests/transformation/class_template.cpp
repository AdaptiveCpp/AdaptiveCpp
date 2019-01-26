class foo {
    template<typename T>
    void bar(T value);
};

template <typename T>
void baz(T) {}

template <typename T>
void foo::bar(T value) {
    baz(value);
}

int main() {
    return 0;
}
