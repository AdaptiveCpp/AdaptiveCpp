
__device__
void builtin_function()
{}

// Should be __host__ __device__
void device_function()
{}

// Should be __host__
void uncalled_function()
{ device_function(); }

// Should be __device__,
// since __device__ builtin is called
void uncalled_builtin_call()
{
  builtin_function();
}

template<class Func>
__global__ void kernel(Func f)
{
  f();
}

int main()
{
  // Should be __device__, since
  // it's passed to a kernel
  auto f = [](){
    device_function();
  };
  
  kernel(f);
}
