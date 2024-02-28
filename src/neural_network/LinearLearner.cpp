//
// Created by Khurram Javed on 2024-02-18.
//

#include "../../include/neural_network/LinearLearner.h"
#include <vector>
#include <iostream>
#include <math.h>

LinearLearner::LinearLearner() {
}

SemiGradientTDLambda::SemiGradientTDLambda(int num_features, float lambda, float meta_step_size, float alpha,
                                           float gamma, float eps) {
    this->lambda = lambda;
    this->theta = meta_step_size; //meta step size
    this->gamma = gamma;

    this->counter = 0;
    this->v = 0;
    this->v_next = 0;
    this->eps = eps;

    this->weights = std::vector<float>(num_features, 0);
    this->features_old = std::vector<float>(num_features, 0);

    this->h = std::vector<float>(num_features, 0);
    this->e = std::vector<float>(num_features, 0);
    this->betas = std::vector<float>(num_features, log(alpha));
    this->idbd_trace = std::vector<float>(num_features, 0);


}

float SemiGradientTDLambda::Step(std::vector<float> &features, float reward) {
    this->counter++;
    this->v = Math::DotProduct(this->weights, features_old);
    this->v_next = Math::DotProduct(this->weights, features);
    float td_error = reward + this->gamma * this->v_next - this->v;

    for (int i = 0; i < features_old.size(); i++) {
//        std::cout << features_old.size() << std::endl;
//        std::cout << i << std::endl;
        this->idbd_trace[i] = this->idbd_trace[i] * this->gamma * this->lambda + this->h[i] * features_old[i];
        this->betas[i] += this->theta / (exp(this->betas[i]) + this->eps) * (td_error * this->idbd_trace[i]);
        float step_size = exp(this->betas[i]);
        this->e[i] = this->e[i] * this->gamma * this->lambda + features_old[i];
        this->weights[i] = this->weights[i] + step_size * (td_error) * this->e[i];
        if ((1 - step_size * features_old[i] * e[i]) > 0)
            this->h[i] = this->h[i] * (1 + step_size * (gamma * features[i] - features_old[i]) * e[i]) +
                         step_size * e[i] * td_error;
        else
            this->h[i] = step_size * e[i] * td_error;
    }
    this->features_old = features;
    return this->v_next;
}

FullGradientTDLambda::FullGradientTDLambda(int num_features, float lambda, float meta_step_size, float alpha,
                                           float gamma, float eps) {
    this->lambda = lambda;
    this->theta = meta_step_size; //meta step size
    this->gamma = gamma;
    this->weights = std::vector<float>(num_features, 0);
    this->features_old = std::vector<float>(num_features, 0);
    this->counter = 0;
    this->v = 0;
    this->v_next = 0;
    this->eps = eps;
    this->h = std::vector<float>(num_features, 0);
    this->e = std::vector<float>(num_features, 0);
    this->betas = std::vector<float>(num_features, log(alpha));
    this->idbd_trace = std::vector<float>(num_features, 0);


    this->h_bar = std::vector<float>(num_features, 0);
    this->y = std::vector<float>(num_features, 0);
    this->u = std::vector<float>(num_features, 0);
}


float FullGradientTDLambda::Step(std::vector<float> &features, float reward) {
    this->counter++;
    this->v = Math::DotProduct(this->weights, features_old);
    this->v_next = Math::DotProduct(this->weights, features);
    float td_error = reward + this->gamma * this->v_next - this->v;
    for (int i = 0; i < features_old.size(); i++) {
        float delta_grad = (gamma * features[i] - features_old[i]);

        h_bar[i] = gamma * gamma * lambda * lambda * h_bar[i] + h[i];
        y[i] = gamma * lambda * y[i] + td_error * h_bar[i];
        betas[i] = betas[i] - theta / (exp(betas[i]) + eps) *
                              (y[i] * delta_grad + gamma * lambda * u[i] * td_error);
        float step_size = exp(betas[i]);
        u[i] = gamma * lambda * u[i] + h_bar[i] * delta_grad;
        e[i] = e[i] * this->gamma * this->lambda + features_old[i];
        weights[i] = weights[i] + step_size * (td_error) * e[i];
        if ((1 - step_size * delta_grad * e[i]) > 0)
            h[i] = h[i] * (1 + step_size * delta_grad * e[i]) + td_error * step_size * e[i];
        else
            h[i] = td_error * step_size * e[i];
    }
    this->features_old = features;
    return this->v_next;
}

SwiftTD::SwiftTD(int num_features, float lambda, float initial_alpha, float gamma, float eps,
                 float max_step_size, float step_size_decay, float meta_step_size,
                 float weight_decay) {
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
//    float td_error_new = reward + this->gamma * value_next - value;
    float dot_product_eligibility = Math::DotProduct(this->e, features_old);
    float scale_factor = 1;

    float sum_of_step_sizes = 0;
    for (int i = 0; i < features.size(); i++) {
        sum_of_step_sizes += exp(this->betas[i]) * features_old[i] * features_old[i];
    }

    if (sum_of_step_sizes > this->max_step_size) {
        scale_factor = this->max_step_size / sum_of_step_sizes;
    }
    for (int i = 0; i < features_old.size(); i++) {
        idbd_trace[i] = idbd_trace[i] * this->gamma * this->lambda + this->h[i] * features_old[i];
        if (scale_factor == 1) {
            this->betas[i] += this->meta_step_size / (exp(this->betas[i]) + eps) * (td_error * idbd_trace[i]);
//            if (this->betas[i] > 0) {
//                this->betas[i] = 0;
//                this->idbd_trace[i] = 0;
//                this->h[i] = 0;
//                this->h_old[i] = 0;
//                this->be[i] = 0;
//            }
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
    }

    this->features_old = features;
    this->v_old = value_next;
    return value_next;
}


std::vector<float> SwiftTD::GetStepSizePerPixel() {
    std::vector<float> data(84 * 84, 0);
    for (int i = 0; i < 84 * 84; i++) {
        for (int j = 0; j < 16; j++) {
            data[i] += exp(this->betas[i * 16 + j]);
        }
    }
    return data;
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