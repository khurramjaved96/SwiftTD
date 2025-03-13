//
// Created by Khurram Javed on 2024-02-18.
//

#include "SwiftTD.h"
#include <vector>
#include <iostream>
#include <math.h>

SwiftTD::SwiftTD() {
}

std::vector<float> SwiftTD::GetWeights() {
    return this->weights;
}


SwiftTDDense::SwiftTDDense(int num_features, float lambda, float initial_alpha, float gamma, float eps,
                                       float max_step_size, float step_size_decay, float meta_step_size) {
    this->beta_normalizer = eps;
    this->eps = step_size_decay;
    this->log_eps = log(step_size_decay);

    this->lambda = lambda;
    this->gamma = gamma;
    this->weights = std::vector<float>(num_features, 0);
    this->betas = std::vector<float>(num_features, log(initial_alpha));
    this->alpha_cache = std::vector<float>(num_features, initial_alpha);
    this->h = std::vector<float>(num_features, 0);
    this->h_old = std::vector<float>(num_features, 0);
    this->h_temp = std::vector<float>(num_features, 0);
    this->z_delta = std::vector<float>(num_features, 0);
    this->z_bar = std::vector<float>(num_features, 0);
    this->p = std::vector<float>(num_features, 0);
    this->z = std::vector<float>(num_features, 0);
    this->delta_w_i = std::vector<float>(num_features, 0);
    this->feature_counter = std::vector<float>(num_features, 0);
    this->v_delta = 0;
    this->counter = 0;
    this->v_old = 0;
    this->v = 0;
    this->meta_step_size = meta_step_size;
    this->eta = max_step_size;
    this->log_eta = log(this->eta);
}


float Math::DotProduct(std::vector<float> &a, std::vector<float> &b) {
    float result = 0;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}


float SwiftTDDense::Step(std::vector<float> &features, float reward) {
    this->counter++;
//    std::vector<float> features(this->weights.size(), 0);
//    for (int i = 0; i < features_temp.size(); i++) {
//        features[features_temp[i]] = 1;
//    }
    float decay = this->gamma * this->lambda;
    this->v = Math::DotProduct(this->weights, features);
    float delta = reward + this->gamma * this->v - this->v_old;
    for (int i = 0; i < features.size(); i++) {
        if (z[i] != 0) {
            this->delta_w_i[i] = delta * z[i] - z_delta[i] * this->v_delta;
            this->weights[i] += this->delta_w_i[i];
            this->betas[i] += this->meta_step_size / (alpha_cache[i]) * delta * p[i];
            if (this->betas[i] > log_eta) {
                this->betas[i] = log_eta;
            }
            if (this->betas[i] < log(beta_normalizer)) {
                this->betas[i] = log(beta_normalizer);
            }
            alpha_cache[i] = exp(this->betas[i]);
            this->h_old[i] = this->h[i];
            this->h[i] = this->h_temp[i];
            this->h_temp[i] = this->h[i] + delta * z_bar[i] - z_delta[i] * this->v_delta;
            this->z_delta[i] = 0;
            this->z[i] *= decay;
            this->p[i] *= decay;
            this->z_bar[i] *= decay;
        }
    }

    this->v_delta = 0;
    float effective_step_size = 0;
    for (int i = 0; i < features.size(); i++) {
        effective_step_size += alpha_cache[i] * features[i] * features[i];
    }

    this->unbounded_rate_of_learning = effective_step_size;
    float E = this->eta;
    if (effective_step_size > this->eta) {
        E = effective_step_size;
    }
    actual_rate_of_learning = (this->eta / E) * unbounded_rate_of_learning;

    float T = 0;
    for (int i = 0; i < features.size(); i++) {
        T += this->z[i] * features[i];
    }
    for (int i = 0; i < features.size(); i++) {
        if (features[i] != 0) {
            this->v_delta += this->delta_w_i[i] * features[i];
            this->z_delta[i] = (this->eta / E) * alpha_cache[i] * features[i];
            this->z[i] += z_delta[i] * (1 - T);
            this->p[i] += features[i] * this->h[i];
            this->z_bar[i] += z_delta[i] * (1 - T - features[i] * this->z_bar[i]);
            this->h_temp[i] = this->h_temp[i] - this->h_old[i] * features[i] * (this->z[i] - this->z_delta[i]) -
                              this->h[i] * this->z_delta[i] * features[i];
            if (effective_step_size > this->eta) {
                this->betas[i] += features[i] * log_eps;
                this->alpha_cache[i] = exp(this->betas[i]);
                if (features[i] != 0) {
                    this->p[i] = 0;
                    this->z_bar[i] = 0;
                    this->h_temp[i] = 0;
                    this->h_old[i] = 0;
                }
            }
        }
    }
    this->v_old = this->v;
    return this->v;
}




void SwiftTD::SetGamma(float gamma) {
    this->gamma = gamma;
}


SwiftTDSparse::SwiftTDSparse(int num_features, float lambda, float initial_alpha, float gamma,
                                                 float eps,
                                                 float max_step_size, float step_size_decay, float meta_step_size) {
    this->beta_normalizer = eps;
    this->eps = step_size_decay;
    this->log_eps = log(step_size_decay);

    this->lambda = lambda;
    this->gamma = gamma;
    this->weights = std::vector<float>(num_features, 0);
    this->betas = std::vector<float>(num_features, log(initial_alpha));
    this->alpha_cache = std::vector<float>(num_features, initial_alpha);
    this->h = std::vector<float>(num_features, 0);
    this->h_old = std::vector<float>(num_features, 0);
    this->h_temp = std::vector<float>(num_features, 0);
    this->z_delta = std::vector<float>(num_features, 0);
    this->z_bar = std::vector<float>(num_features, 0);
    this->p = std::vector<float>(num_features, 0);
    this->z = std::vector<float>(num_features, 0);
    this->delta_w_i = std::vector<float>(num_features, 0);
    this->feature_counter = std::vector<float>(num_features, 0);
    this->v_delta = 0;
    this->counter = 0;
    this->v_old = 0;
    this->v = 0;
    this->meta_step_size = meta_step_size;
    this->eta = max_step_size;
    this->log_eta = log(this->eta);
}


float SwiftTDSparse::Step(std::vector<int> &features, float reward) {
    this->counter++;
    float decay = this->gamma * this->lambda;
    this->v = 0;
    for (auto &i: features) {
        this->v += this->weights[i];
    }
    float delta = reward + this->gamma * this->v - this->v_old;
    int j = 0;
    while (j < active_indices.size()) {
        int i = active_indices[j];
        this->delta_w_i[i] = delta * z[i] - z_delta[i] * this->v_delta;
        this->weights[i] += this->delta_w_i[i];
        this->betas[i] += this->meta_step_size / (alpha_cache[i]) * delta * p[i];
        if (this->betas[i] > log_eta) {
            this->betas[i] = log_eta;
        }
        if (this->betas[i] < log(beta_normalizer)) {
            this->betas[i] = log(beta_normalizer);
        }
        alpha_cache[i] = exp(this->betas[i]);
        this->h_old[i] = this->h[i];
        this->h[i] = this->h_temp[i];
        this->h_temp[i] = this->h[i] + delta * z_bar[i] - z_delta[i] * this->v_delta;
        this->z_delta[i] = 0;
        this->z[i] *= decay;
        this->p[i] *= decay;
        this->z_bar[i] *= decay;
        if (z[i] < alpha_cache[i] * 0.01) {
            z[i] = 0;
            p[i] = 0;
            z_bar[i] = 0;
            active_indices[j] = active_indices[active_indices.size() - 1];
            active_indices.pop_back();
        } else {
            j++;
        }
    }

    this->v_delta = 0;
    float effective_step_size = 0;
    for (auto &i: features) {
        effective_step_size += alpha_cache[i];
    }

    this->unbounded_rate_of_learning = effective_step_size;
    float E = this->eta;
    if (effective_step_size > this->eta) {
        E = effective_step_size;
    }
    actual_rate_of_learning = (this->eta / E) * unbounded_rate_of_learning;

    float T = 0;
    for (auto &i: features) {
        T += this->z[i];
    }
    for (auto &i: features) {
        if (z[i] == 0) {
            active_indices.push_back(i);
        }
        this->v_delta += this->delta_w_i[i];
        this->z_delta[i] = (this->eta / E) * alpha_cache[i];
        this->z[i] += z_delta[i] * (1 - T);
        this->p[i] += this->h[i];
        this->z_bar[i] += z_delta[i] * (1 - T - this->z_bar[i]);
        this->h_temp[i] = this->h_temp[i] - this->h_old[i] * (this->z[i] - this->z_delta[i]) -
                          this->h[i] * this->z_delta[i];
        if (effective_step_size > this->eta) {
            this->betas[i] += log_eps;
            this->alpha_cache[i] = exp(this->betas[i]);
            this->z_bar[i] = 0;
            this->h_temp[i] = 0;
            this->h_old[i] = 0;
        }


    }
    this->v_old = this->v;
    return this->v;
}
