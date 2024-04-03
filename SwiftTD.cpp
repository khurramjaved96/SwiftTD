//
// Created by Khurram Javed on 2024-02-18.
//

#include "SwiftTD.h"
#include <vector>
#include <math.h>

LinearLearner::LinearLearner() {
}

std::vector<float> LinearLearner::GetWeights() {
    return this->weights;
}

SwiftTD::SwiftTD(int num_features, float lambda, float initial_alpha, float gamma, float eps,
                 float max_step_size, float step_size_decay, float meta_step_size) {
    this->step_size_decay = step_size_decay;
    this->lambda = lambda;
    this->gamma = gamma;
    this->eps = eps;
    this->weights = std::vector<float>(num_features, 0);
    this->betas = std::vector<float>(num_features, log(initial_alpha));
    this->h = std::vector<float>(num_features, 0);
    this->h_old = std::vector<float>(num_features, 0);
    this->be = std::vector<float>(num_features, 0);
    this->idbd_trace = std::vector<float>(num_features, 0);
    this->e = std::vector<float>(num_features, 0);
    this->momentum_term = std::vector<float>(num_features, 0);
    this->gradient_norm = std::vector<float>(num_features, 0);
    this->features_old = std::vector<float>(num_features, 0);
    this->feature_counter = std::vector<float>(num_features, 0);
    this->counter = 0;
    this->v_old = 0;
    this->v = 0;
    this->meta_step_size = meta_step_size;
    this->max_step_size = max_step_size;
}


float Math::DotProduct(std::vector<float> &a, std::vector<float> &b) {
    float result = 0;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}


float SwiftTD::Step(std::vector<float> &features, float reward) {
    this->counter++;
    float value = Math::DotProduct(this->weights, features_old);
    float value_next = Math::DotProduct(this->weights, features);
    float td_error = reward + this->gamma * value_next - this->v_old;
    float dot_product_eligibility = Math::DotProduct(this->e, features_old);
    float scale_factor = 1;

    float sum_of_step_sizes = 0;
    for (int i = 0; i < features.size(); i++) {
        float temp = exp(this->betas[i]) * features_old[i] * features_old[i];
        this->feature_counter[i] += temp;
        sum_of_step_sizes += temp;
    }

    if (sum_of_step_sizes > this->max_step_size) {
        scale_factor = this->max_step_size / sum_of_step_sizes;
    }
    for (int i = 0; i < features_old.size(); i++) {
        idbd_trace[i] = idbd_trace[i] * this->gamma * this->lambda + this->h[i] * features_old[i];

        float step_size = exp(this->betas[i]) * scale_factor;
        this->be[i] = this->lambda * this->gamma * this->be[i] + step_size * features_old[i] -
                      this->gamma * this->lambda * step_size * dot_product_eligibility * features_old[i] -
                      step_size * this->gamma * this->lambda * features_old[i] * features_old[i] * this->be[i];

        this->e[i] = this->e[i] * this->gamma * this->lambda;
        this->e[i] = this->e[i] + features_old[i] * step_size -
                     step_size * this->gamma * this->lambda * (dot_product_eligibility) * features_old[i];

        this->weights[i] =
                this->weights[i] + (td_error) * this->e[i] - step_size * (value - this->v_old) * features_old[i];

        float temp = this->h[i];

        this->h[i] = this->h[i] + td_error * this->be[i] +
                     this->e[i] * (-1 * this->h_old[i] * features_old[i])
                     - features_old[i] *
                       (step_size * (value - this->v_old) +
                        step_size * features_old[i] * (this->h[i] - this->h_old[i]));
        this->h_old[i] = temp;
        if (scale_factor == 1) {
            this->betas[i] += this->meta_step_size / (exp(this->betas[i]) + eps) * (td_error * idbd_trace[i]);
            if (this->betas[i] > 0) {
                this->betas[i] = 0;
            }
        }
        if (scale_factor < 1) {
            if (features_old[i] > 0) {
                this->betas[i] += log(step_size_decay);
                this->idbd_trace[i] = 0;
                this->h[i] = 0;
                this->h_old[i] = 0;
                this->be[i] = 0;
            }
        }
    }

    this->features_old = features;
    this->v_old = value_next;
    return value_next;
}


void LinearLearner::SetGamma(float gamma) {
    this->gamma = gamma;
}

std::vector<float> SwiftTD::GetStepSizePerPixel() {
    return this->feature_counter;
}

TrueOnlineTDLambda::TrueOnlineTDLambda(int num_features, float lambda, float alpha, float gamma) {
    this->lambda = lambda;
    this->gamma = gamma;
    this->weights = std::vector<float>(num_features, 0);
    this->features_old = std::vector<float>(num_features, 0);
    this->counter = 0;
    this->v_old = 0;
    this->v = 0;
    this->step_size = alpha;
    this->e = std::vector<float>(num_features, 0);
}

float TrueOnlineTDLambda::Step(std::vector<float> &features, float reward) {
    this->counter++;
    float value = Math::DotProduct(this->weights, features_old);
    float value_next = Math::DotProduct(this->weights, features);
    float td_error = reward + this->gamma * value_next - this->v_old;
    float dot_product_eligibility = Math::DotProduct(this->e, features_old);
    for (int i = 0; i < features_old.size(); i++) {
        this->e[i] = this->e[i] * this->gamma * this->lambda;
        this->e[i] = this->e[i] + features_old[i] * step_size -
                     step_size * this->gamma * this->lambda * (dot_product_eligibility) * features_old[i];
        this->weights[i] =
                this->weights[i] + (td_error) * this->e[i] - step_size * (value - this->v_old) * features_old[i];
    }
    this->features_old = features;
    this->v_old = value_next;
    return value_next;
}


