#include <memory>
#include <cassert>
#include <iostream>

#include <CL/sycl.hpp>

using data_type = float;

class device_context
{
public:
  device_context (
    size_t size_arg,
    const data_type * a,
    const data_type * b,
    const data_type * c,
    cl::sycl::device &device)
  : size (size_arg)
  , queue (device)
  , buff_a (a, size)
  , buff_b (b, size)
  , buff_c (c, size)
  {
    assert(a.size() == b.size());
  }

  const size_t size {};
  cl::sycl::queue queue;
  cl::sycl::buffer<data_type> buff_a;
  cl::sycl::buffer<data_type> buff_b;
  cl::sycl::buffer<data_type> buff_c;
};

void add(std::vector<std::unique_ptr<device_context>> &contexts)
{
  for (auto &context: contexts)
    {
      cl::sycl::range<1> work_items{context->size};

      auto e = context->queue.submit([&](cl::sycl::handler& cgh){
        auto access_a = context->buff_a.get_access<cl::sycl::access::mode::read>(cgh);
        auto access_b = context->buff_b.get_access<cl::sycl::access::mode::read>(cgh);
        auto access_c = context->buff_c.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class vector_add>(work_items,
                                           [=] (cl::sycl::id<1> tid) {
                                             access_c[tid] = access_a[tid] + access_b[tid];
                                           });
      });
    }
}

int main()
{
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices ();

  const unsigned int n = 100'000'000;

  std::vector<data_type> a (n, 1);
  std::vector<data_type> b (n, 2);
  std::vector<data_type> c (n);

  double single_gpu_elapsed_time {};

  for (unsigned int devices_count = 1; devices_count <= devices.size (); devices_count++)
    {
      std::vector<std::unique_ptr<device_context>> contexts;
      const unsigned int chunk_size = n / devices_count;

      for (unsigned int dev_id = 0; dev_id < devices_count; dev_id++)
        {
          const unsigned int beg = chunk_size * dev_id;
          const unsigned int end = dev_id == devices_count - 1 ? n : chunk_size * (dev_id + 1);
          const unsigned int size = end - beg;

          contexts.push_back (std::make_unique<device_context> (
            size,
            a.data () + beg,
            b.data () + beg,
            c.data () + beg,
            devices[dev_id]));
        }

      auto begin = std::chrono::system_clock::now (); // TODO Implement event profiler
      add(contexts);
      auto end = std::chrono::system_clock::now ();
      const std::chrono::duration<double> duration = end - begin;
      std::cout << duration.count () << "s on " << devices_count << " devices";

      if (devices_count == 1)
        single_gpu_elapsed_time = duration.count ();
      else
        std::cout << " (speedup = " << single_gpu_elapsed_time / duration.count () << ")";
      std::cout << "\n";
    }
}