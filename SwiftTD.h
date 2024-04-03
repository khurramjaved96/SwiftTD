//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef SWIFTTD_H
#define SWIFTTD_H

#include <vector>

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

    virtual float Step(std::vector<float> &features, float reward) = 0;
};


class TrueOnlineTDLambda : public LinearLearner {
private:
    int counter;
    std::vector<float> features_old;
    float lambda;
    float step_size;
    float v_old;
    float v;
    std::vector<float> e;

public:
    TrueOnlineTDLambda(int num_features, float lambda, float alpha, float gamma);

    float Step(std::vector<float> &features, float reward) override;
};


class SwiftTD : public LinearLearner {
private:
    int counter;
    std::vector<float> features_old;
    std::vector<float> momentum_term;
    std::vector<float> gradient_norm;
    std::vector<float> betas;
    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> be;
    std::vector<float> idbd_trace;
    std::vector<float> e;
    std::vector<float> feature_counter;
    float meta_step_size;
    float lambda;
    float v_old;
    float v;
    float eps;
    float max_step_size;
    float step_size_decay;

public:

    SwiftTD(int num_features, float lambda, float initial_alpha, float gamma, float eps, float max_step_size,
            float step_size_decay, float meta_step_size);

    float Step(std::vector<float> &features, float reward) override;

    std::vector<float> GetStepSizePerPixel();
};

#endif //SWIFTTD_H
