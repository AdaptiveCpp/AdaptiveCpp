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
#ifndef MODEL_HPP
#define MODEL_HPP

#include <random>
#include <vector>
#include <array>
#include "bruteforce_nbody.hpp"

class random_particle_cloud
{
public:

  random_particle_cloud(const vector_type& position,
                        const vector_type& width,
                        arithmetic_type mean_mass,
                        arithmetic_type mass_distribution_width,
                        const vector_type& mean_velocity,
                        const vector_type& velocity_distribution_width)
    : _generator{generate_seed()}
  {
    _means[0] = position.x();
    _means[1] = position.y();
    _means[2] = position.z();
    _stddevs[0] = width.x();
    _stddevs[1] = width.y();
    _stddevs[2] = width.z();

    _means[3] = mean_mass;
    _stddevs[3] = mass_distribution_width;

    _means[4] = mean_velocity.x();
    _means[5] = mean_velocity.y();
    _means[6] = mean_velocity.z();

    _stddevs[4] = velocity_distribution_width.x();
    _stddevs[5] = velocity_distribution_width.y();
    _stddevs[6] = velocity_distribution_width.z();
  }

  void sample(std::size_t n,
              std::vector<particle_type>& particles,
              std::vector<vector_type>& velocities)
  {
    particles.resize(n);
    velocities.resize(n);

    std::array<std::normal_distribution<arithmetic_type>, 7> distributions;
    for(std::size_t i = 0; i < distributions.size(); ++i)
    {
      distributions[i] = std::normal_distribution<arithmetic_type>{
          _means[i],
          _stddevs[i]
      };
    }

    for(std::size_t i = 0; i < n; ++i)
    {
      particle_type sampled_particle{
        distributions[0](_generator),
        distributions[1](_generator),
        distributions[2](_generator),
        distributions[3](_generator),
      };

      sampled_particle.w() = std::abs(sampled_particle.w());
      particles[i] = sampled_particle;

      vector_type sampled_velocity{
        distributions[4](_generator),
        distributions[5](_generator),
        distributions[6](_generator)
      };

      velocities[i] = sampled_velocity;
    }
  }

private:
  static std::size_t generate_seed()
  {
    std::random_device rd;
    return rd();
  }

  std::array<arithmetic_type, 7> _means;
  std::array<arithmetic_type, 7> _stddevs;

  std::mt19937 _generator;
};

#endif
