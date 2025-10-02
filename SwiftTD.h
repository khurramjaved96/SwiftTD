
//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef SWIFTTD_H
#define SWIFTTD_H

#include <vector>

class Math
{
public:
    static float DotProduct(const std::vector<float>& a, const std::vector<float>& b);
};

class SwiftTDDense
{
private:
    std::vector<int> setOfEligibleItems; // set of eligible items
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;

    std::vector<float> featureVector;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::vector<float> last_alpha;


    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;

    float decay;
    float gamma;

public:
    SwiftTDDense(int num_features, float lambda, float initial_alpha, float gamma, float eps, float max_step_size,
                 float step_size_decay, float meta_step_size);
    float Step(const std::vector<float>& features, float reward);
};

class SwiftTDSparseAndBinaryFeatures
{
    std::vector<int> setOfEligibleItems; // set of eligible items
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;

    std::vector<float> featureVector;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::vector<float> last_alpha;


    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;

    float decay;
    float gamma;

public:
    SwiftTDSparseAndBinaryFeatures(int num_features, float lambda, float alpha, float gamma, float epsilon,
                                   float meta_step_size,
                                   float eta, float decay);
    float Step(const std::vector<int>& feature_indices, float reward);
};


class SwiftTDSparseAndNonBinaryFeatures
{
    std::vector<std::pair<int, float>> setOfEligibleItems; // set of eligible items
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;

    std::vector<float> featureVector;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::vector<float> last_alpha;


    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;

    float decay;
    float gamma;

public:
    SwiftTDSparseAndNonBinaryFeatures(int num_features, float lambda, float alpha, float gamma, float epsilon,
                                      float meta_step_size,
                                      float eta, float decay);
    float Step(const std::vector<std::pair<int, float>>& feature_indices, float reward);
};

#endif // SWIFTTD_H
