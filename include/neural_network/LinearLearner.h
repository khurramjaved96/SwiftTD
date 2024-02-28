//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
#define TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
#include <vector>

class Math{
public:
    static float DotProduct(std::vector<float> &a, std::vector<float> &b);
};

class LinearLearner{
public:
    LinearLearner();
    virtual float Step(std::vector<float> &features, float reward) = 0;
};

class SemiGradientTDLambda : public LinearLearner{
private:
    float change_in_value;
    float dot_product;
    int counter;
    std::vector<float> features_old;
    std::vector<float> weights;
    float alpha;
    float gamma;
    float lambda;
    float v_old;
    float v;


    std::vector<float> features_old;
    std::vector<float> weights;
    std::vector<float> betas;
    std::vector<float> h;
    std::vector<float> e;


public:
    SemiGradientTDLambda(int num_features, float lambda, float alpha, float gamma);
    float Step(std::vector<float> &features, float reward) override;
};



class TrueOnlineTDLambda : public LinearLearner{
private:
    float change_in_value;
    float dot_product;
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
    float beta_1;
    float beta_2;
    float lambda;
    float gamma;
    float v_old;
    float v;
    float eps;
    float rho;
    float max_step_size;
    float step_size_decay;
    float weight_decay;

public:

    TrueOnlineTDLambda(int num_features, float lambda, float initial_alpha, float gamma, float eps, float max_step_size, float step_size_decay, float meta_step_size, float weight_decay);
    float Step(std::vector<float> &features, float reward) override;
    void DecayEligibility();
    void ResetTrace();
    void SetWeights(std::vector<float> weights);
    std::vector<float> GetWeights();
    void SetBetas(std::vector<float> betas);
    std::vector<float> GetBetas();
    std::vector<float> GetH();
    std::vector<float> GetStepSizePerPixel();
};
#endif //TRUEONLINETDLAMBDATEST_LINEARLEARNER_H
