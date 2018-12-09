

template<class T>
__device__
T builtin_function()
{ return T{}; }

template<class T>
class device_class
{
public:
  device_class()
  : _x{builtin_function<T>()}
  {}
  
  T member() { return _x + builtin_function<T>(); }
  
  template<class Y>
  Y template_member(T x)
  { return x + static_cast<T>(Y{}); }
  
private:
  T _x;
};

template<class T>
class unused_device_class
{
public:
  unused_device_class()
  : _x{builtin_function<float>()}
  {}
  
   T member() { return _x + builtin_function<float>(); }
  
  template<class Y>
   Y template_member(T x)
  { return x + static_cast<T>(builtin_function<float>()); }
  
private:
  T _x;
};


template<class Func>
__global__ void kernel(Func f)
{
  f();
}

// Should be __host__
template<class Func>
void submit(Func f)
{
  f();
}

template<class T>
void templated_launch()
{
  submit([](){
    
    kernel([](){
      device_class<T> d;
      d.member();
      d.template template_member<float>(T{});
    });
    
  });
}


int main()
{
  templated_launch<int>();
  templated_launch<float>();
}
