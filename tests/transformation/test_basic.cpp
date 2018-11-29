
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

__global__ void kernel()
{
  device_function();
}

int main()
{
  kernel();
}
