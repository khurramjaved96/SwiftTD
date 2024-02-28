//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
#define TRUEONLINETDLAMBDATEST_LINEARLEARNER_H

#include <vector>

class Math {
public:
    static float DotProduct(std::vector<float> &a, std::vector<float> &b);
};

class LinearLearner {
public:
    LinearLearner();

    virtual float Step(std::vector<float> &features, float reward) = 0;
};

class SemiGradientTDLambda : public LinearLearner {
private:
    int counter;
    std::vector<float> features_old;
    std::vector<float> weights;
    float gamma;
    float lambda;
    float v;
    float v_next;
    float theta;
    float eps;
    std::vector<float> betas;
    std::vector<float> h;
    std::vector<float> e;
    std::vector<float> idbd_trace;
public:
    SemiGradientTDLambda(int num_features, float lambda, float meta_step_size, float alpha, float gamma, float eps);

    float Step(std::vector<float> &features, float reward) override;
};

class FullGradientTDLambda : public LinearLearner {
private:
    int counter;
    std::vector<float> features_old;
    std::vector<float> weights;
    float gamma;
    float lambda;
    float v;
    float v_next;
    float theta;
    float eps;
    std::vector<float> betas;
    std::vector<float> h;
    std::vector<float> h_bar;
    std::vector<float> y;
    std::vector<float> u;
    std::vector<float> e;
    std::vector<float> idbd_trace;

public:
    FullGradientTDLambda(int num_features, float lambda, float meta_step_size, float alpha, float gamma, float eps);

    float Step(std::vector<float> &features, float reward) override;
};


class TrueOnlineTDLambda : public LinearLearner {
private:
    int counter;
    std::vector<float> features_old;
    std::vector<float> weights;
    float gamma;
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
    std::vector<float> weights;
    std::vector<float> betas;
    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> be;
    std::vector<float> idbd_trace;
    std::vector<float> e;
    float meta_step_size;
    float lambda;
    float gamma;
    float v_old;
    float v;
    float eps;
    float max_step_size;
    float step_size_decay;

public:

    SwiftTD(int num_features, float lambda, float initial_alpha, float gamma, float eps, float max_step_size,
            float step_size_decay, float meta_step_size, float weight_decay);

    float Step(std::vector<float> &features, float reward) override;

    std::vector<float> GetStepSizePerPixel();
};

#endif //TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
