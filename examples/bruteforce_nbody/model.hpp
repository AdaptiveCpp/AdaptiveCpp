/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
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
  static typename std::mt19937::result_type generate_seed()
  {
    std::random_device rd;
    return rd();
  }

  std::array<arithmetic_type, 7> _means;
  std::array<arithmetic_type, 7> _stddevs;

  std::mt19937 _generator;
};

#endif
