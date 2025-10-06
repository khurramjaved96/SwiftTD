
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

class SwiftTDNonSparse
{
private:
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

    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;

    float decay;
    float gamma;

public:
    SwiftTDNonSparse(int number_of_features, float lambda_init, float alpha_init, float gamma_init, float epsilon_init,
                     float eta_init,
                     float decay_init, float meta_step_size_init);
    float Step(const std::vector<float>& features, float reward);
};

class SwiftTDBinaryFeatures
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
    SwiftTDBinaryFeatures(int number_of_features, float lambda_init, float alpha_init, float gamma_init,
                          float epsilon_init, float eta_init,
                          float decay_init, float meta_step_size_init);
    float Step(const std::vector<int>& feature_indices, float reward);
};


class SwiftTD
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
    SwiftTD(int num_features, float lambda_, float initial_alpha, float gamma_,
            float epsilon_,
            float eta_, float step_size_decay_, float meta_step_size_);
    float Step(const std::vector<std::pair<int, float>>& feature_indices, float reward);
};

#endif // SWIFTTD_H
