//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef SWIFTTD_H
#define SWIFTTD_H

#include <vector>

class Math
{
public:
    static float DotProduct(const std::vector<float> &a, const std::vector<float> &b);
};

class SwiftTDDense 
{
private:
    std::vector<float> weights;
    int counter;
    float gamma;
    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> z_delta;
    std::vector<float> delta_w_i;
    std::vector<float> z_bar;
    std::vector<float> p;
    std::vector<float> z;
    std::vector<float> feature_counter;
    std::vector<float> alpha_cache;
    float log_eta;
    float log_eps;
    float v_delta;
    float beta_normalizer;
    float meta_step_size;
    float lambda;
    float v_old;
    float v;
    float eta;
    float eps;
    std::vector<float> betas;
    float unbounded_rate_of_learning;
    float actual_rate_of_learning;
public:
    SwiftTDDense(int num_features, float lambda, float initial_alpha, float gamma, float eps, float max_step_size,
                 float step_size_decay, float meta_step_size);
    float Step(const std::vector<float> &features, float reward);
};

class SwiftTDSparse
{
private:
    std::vector<float> weights;
    int counter;
    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> z_delta;
    std::vector<float> delta_w_i;
    std::vector<float> z_bar;
    std::vector<float> p;
    std::vector<float> z;
    std::vector<float> feature_counter;
    std::vector<float> alpha_cache;
    float log_eta;
    float log_eps;
    float gamma;
    float v_delta;
    float beta_normalizer;
    float meta_step_size;
    float lambda;
    float v_old;
    float v;
    float eta;
    std::vector<int> active_indices;
    float eps;
    std::vector<float> betas;
    float unbounded_rate_of_learning;
    float actual_rate_of_learning;
public:

    SwiftTDSparse(int num_features, float lambda, float initial_alpha, float gamma, float eps,
                  float max_step_size,
                  float step_size_decay, float meta_step_size);
    float Step(const std::vector<int> &features_indices, float reward);
};

#endif // SWIFTTD_H
