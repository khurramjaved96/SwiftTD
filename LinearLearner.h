//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
#define TRUEONLINETDLAMBDATEST_LINEARLEARNER_H

#include <vector>
#include <random>
class Math {
public:
    static float DotProduct(std::vector<float> &a, std::vector<float> &b);
};

class LinearLearner {
protected:
    float gamma;
    std::vector<float> weights;
public:
    LinearLearner();

    std::vector<float> GetWeights();

    void SetGamma(float gamma);

    virtual std::vector<float> GetStepSizePerPixel();
};


class SwiftTDReorganized : public LinearLearner {
private:
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
    float v_delta;
    float beta_normalizer;
    float meta_step_size;
    float lambda;
    float v_old;
    float v;
    float eta;

    float eps;

public:
    std::vector<float> betas;
    float unbounded_rate_of_learning;
    float actual_rate_of_learning;

    SwiftTDReorganized(int num_features, float lambda, float initial_alpha, float gamma, float eps, float max_step_size,
                       float step_size_decay, float meta_step_size);

    float Step(std::vector<float> &features_indices, float reward);

    std::vector<float> GetStepSizePerPixel() override;
};


class SwiftTDReorganizedCache : public LinearLearner {
private:
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
    float v_delta;
    float beta_normalizer;
    float meta_step_size;
    float lambda;
    float v_old;
    float v;
    float eta;
    std::vector<int> active_indices;

    float eps;

public:
    std::vector<float> betas;
    float unbounded_rate_of_learning;
    float actual_rate_of_learning;

    SwiftTDReorganizedCache(int num_features, float lambda, float initial_alpha, float gamma, float eps,
                            float max_step_size,
                            float step_size_decay, float meta_step_size);

    float Step(std::vector<int> &features_indices, float reward);

    std::vector<float> GetStepSizePerPixel() override;
};


#endif //TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
