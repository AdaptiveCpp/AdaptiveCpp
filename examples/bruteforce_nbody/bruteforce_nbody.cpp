/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "bruteforce_nbody.hpp"
#include "model.hpp"


arithmetic_type mirror_position(const arithmetic_type mirror_pos,
                                const arithmetic_type position)
{
  arithmetic_type delta = cl::sycl::fabs(mirror_pos - position);
  return (position <= mirror_pos) ?
        mirror_pos + delta : mirror_pos - delta;
}

int get_num_iterations_per_output_step()
{
  char* val = std::getenv("NBODY_ITERATIONS_PER_OUTPUT");
  if(!val)
    return 1;
  return std::stoi(val);
}

template<class T, int Dim>
using local_accessor =
  sycl::accessor<T,Dim,
                 sycl::access::mode::read_write,
                 sycl::access::target::local>;

int main()
{
  const int iterations_per_output =
      get_num_iterations_per_output_step();

  std::vector<particle_type> particles;
  std::vector<vector_type> velocities;

  arithmetic_type particle_mass = total_mass / num_particles;

  random_particle_cloud particle_distribution0{
    vector_type{0.0f, 100.0f, 0.0f},
    vector_type{10.0f, 15.0f, 11.0f},
    particle_mass, 0.1f * particle_mass,
    vector_type{0.0f, -26.0f, 5.0f},
    vector_type{5.0f, 20.0f, 12.f}
  };


  random_particle_cloud particle_distribution1{
    vector_type{50.0f, 5.0f, 0.0f},
    vector_type{17.0f, 7.0f, 5.0f},
    particle_mass, 0.1f * particle_mass,
    vector_type{-5.f, 20.0f, 1.0f},
    vector_type{18.0f, 11.f, 8.f}
  };

  random_particle_cloud particle_distribution2{
    vector_type{-50.0f, -100.0f, 0.0f},
    vector_type{10.0f, 10.0f, 14.0f},
    particle_mass, 0.1f * particle_mass,
    vector_type{5.f, 5.0f, -1.0f},
    vector_type{10.0f, 6.f, 5.f}
  };

  particle_distribution0.sample(0.2 * num_particles,
                                particles, velocities);

  std::vector<particle_type> particles_cloud1, particles_cloud2;
  std::vector<vector_type> velocities_cloud1, velocities_cloud2;
  particle_distribution1.sample(0.4 * num_particles,
                                particles_cloud1, velocities_cloud1);

  particle_distribution2.sample(0.4 * num_particles,
                                particles_cloud2, velocities_cloud2);

  particles.insert(particles.end(),
                   particles_cloud1.begin(),
                   particles_cloud1.end());

  particles.insert(particles.end(),
                   particles_cloud2.begin(),
                   particles_cloud2.end());

  velocities.insert(velocities.end(),
                   velocities_cloud1.begin(),
                   velocities_cloud1.end());

  velocities.insert(velocities.end(),
                   velocities_cloud2.begin(),
                   velocities_cloud2.end());

  auto particles_buffer =
      sycl::buffer<particle_type, 1>{particles.data(), particles.size()};
  auto velocities_buffer =
      sycl::buffer<vector_type, 1>{velocities.data(), velocities.size()};
  auto forces_buffer =
      sycl::buffer<vector_type, 1>{sycl::range<1>{particles.size()}};

  sycl::default_selector selector;
  sycl::queue q{selector};

  auto execution_range = sycl::nd_range<1>{
      sycl::range<1>{((num_particles + local_size - 1) / local_size) * local_size},
      sycl::range<1>{local_size}
  };


  std::ofstream outputfile{"output.txt"};
  for(std::size_t t = 0; t < num_timesteps; ++t)
  {
    // Submit force calculation
    q.submit([&](sycl::handler& cgh){
      auto particles_access =
          particles_buffer.get_access<sycl::access::mode::read>(cgh);
      auto forces_access =
          forces_buffer.get_access<sycl::access::mode::discard_write>(cgh);

      auto scratch = local_accessor<particle_type, 1>{
        sycl::range<1>{local_size},
        cgh
      };

      cgh.parallel_for<class force_calculation_kernel>(execution_range,
                                                       [=](sycl::nd_item<1> tid){
        const size_t global_id = tid.get_global_id().get(0);
        const size_t local_id = tid.get_local_id().get(0);
        const size_t num_particles = particles_access.get_range()[0];
        vector_type force{0.0f};

        const particle_type my_particle =
            (global_id < num_particles) ? particles_access[global_id] : particle_type{0.0f};

        for(size_t offset = 0; offset < num_particles; offset += local_size)
        {
          if(global_id < num_particles)
            scratch[local_id] = particles_access[offset + local_id];
          else
            scratch[local_id] = particle_type{0.0f};
          tid.barrier();

          for(int i = 0; i < local_size; ++i)
          {
            const particle_type p = scratch[i];
            const vector_type p_direction = p.swizzle<0,1,2>();
            const vector_type R = p_direction - my_particle.swizzle<0,1,2>();
            // dot is not yet implemented
            const arithmetic_type r_inv =
                sycl::rsqrt(R.x()*R.x() + R.y()*R.y() + R.z()*R.z()
                                    + gravitational_softening);

            // Actually we just calculate the acceleration, not the
            // force. We only need the acceleration anyway.
            if(global_id != offset + i)
              force += static_cast<arithmetic_type>(p.w()) * r_inv * r_inv * r_inv * R;
          }

          tid.barrier();
        }

        if(global_id < num_particles)
          forces_access[global_id] = force;
      });
    });

    // Time integration
    q.submit([&](cl::sycl::handler& cgh){
      auto particles_access =
          particles_buffer.get_access<sycl::access::mode::read_write>(cgh);
      auto velocities_access =
          velocities_buffer.get_access<sycl::access::mode::read_write>(cgh);
      auto forces_access =
          forces_buffer.get_access<sycl::access::mode::read>(cgh);
      const arithmetic_type dt = ::dt;

      cgh.parallel_for<class integration_kernel>(execution_range,
                                                [=](sycl::nd_item<1> tid){
        const size_t global_id = tid.get_global_id().get(0);
        const size_t num_particles = particles_access.get_range().get(0);

        if(global_id < num_particles)
        {
          particle_type p = particles_access[global_id];
          vector_type v = velocities_access[global_id];
          const vector_type acceleration = forces_access[global_id];

          // Bring v to the current state
          v += acceleration * dt;

          // Update position
          p.x() += v.x() * dt;
          p.y() += v.y() * dt;
          p.z() += v.z() * dt;

          // Reflect particle position and invert velocities
          // if particles exit the simulation cube
          if(static_cast<arithmetic_type>(p.x()) <= -half_cube_size)
          {
            v.x() = cl::sycl::fabs(v.x());
            p.x() = mirror_position(-half_cube_size, p.x());
          }
          else if(static_cast<arithmetic_type>(p.x()) >= half_cube_size)
          {
            v.x() = -cl::sycl::fabs(v.x());
            p.x() = mirror_position(half_cube_size, p.x());
          }

          if(static_cast<arithmetic_type>(p.y()) <= -half_cube_size)
          {
            v.y() = cl::sycl::fabs(v.y());
            p.y() = mirror_position(-half_cube_size, p.y());
          }
          else if(static_cast<arithmetic_type>(p.y()) >= half_cube_size)
          {
            v.y() = -cl::sycl::fabs(v.y());
            p.y() = mirror_position(half_cube_size, p.y());
          }

          if(static_cast<arithmetic_type>(p.z()) <= -half_cube_size)
          {
            v.z() = cl::sycl::fabs(v.z());
            p.z() = mirror_position(-half_cube_size, p.z());
          }
          else if(static_cast<arithmetic_type>(p.z()) >= half_cube_size)
          {
            v.z() = -cl::sycl::fabs(v.z());
            p.z() = mirror_position(half_cube_size, p.z());
          }

          particles_access[global_id] = p;
          velocities_access[global_id] = v;
        }
      });
    });

    if(t % iterations_per_output == 0)
    {
      std::cout << "Writing output..."  << std::endl;
      auto particle_positions =
          particles_buffer.get_access<sycl::access::mode::read>();

      for(std::size_t i = 0; i < num_particles; ++i)
      {
        outputfile << particle_positions[i].x() << " "
                   << particle_positions[i].y() << " "
                   << particle_positions[i].z() << " " << i << std::endl;
      }
    }
  }
}
