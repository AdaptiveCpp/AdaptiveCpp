#include <memory>
#include <thread>
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


void add(device_context &context)
{
  cl::sycl::range<1> work_items{context.size};

  context.queue.submit([&](cl::sycl::handler& cgh){
    auto access_a = context.buff_a.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_b = context.buff_b.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_c = context.buff_c.get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<class vector_add>(work_items,
                                       [=] (cl::sycl::id<1> tid) {
                                         access_c[tid] = access_a[tid] + access_b[tid];
                                       });
  });
}

int main()
{
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices ();
  std::vector<std::thread> workers;

  const unsigned int n = 100'000'000;
  std::vector<data_type> a (n, 1);
  std::vector<data_type> b (n, 2);
  std::vector<data_type> c (n);

  for (unsigned int dev_id = 0; dev_id < devices.size (); dev_id++)
    {
      const unsigned int chunk_size = n / devices.size ();
      const unsigned int beg = chunk_size * dev_id;
      const unsigned int end = dev_id == devices.size () - 1 ? n : chunk_size * (dev_id + 1);
      const unsigned int size = end - beg;
      auto &dev = devices[dev_id];

      workers.emplace_back ([&] () {
        device_context context (
          size,
          a.data () + beg,
          b.data () + beg,
          c.data () + beg,
          dev);
        add (context);
      });
    }

  for (auto &worker: workers)
    worker.join ();
}