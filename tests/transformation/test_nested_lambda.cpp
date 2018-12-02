
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

// Should be __host__
template<class Func>
void submit(Func f)
{
  f();
}

int main()
{
  
  submit([](){
    
    kernel([](){
      device_function();
    });
    
  });
}
