//
// Created by Khurram Javed on 2024-02-18.
//

#include "SwiftTD.h"
#include <vector>
#include <math.h>


SwiftTDNonSparse::SwiftTDNonSparse(int number_of_features, float lambda_init, float alpha_init, float gamma_init,
                                   float epsilon_init, float eta_init,
                                   float decay_init, float meta_step_size_init)
{
    this->gamma = gamma_init;
    this->w = std::vector<float>(number_of_features, 0.0f);
    this->featureVector = std::vector<float>(number_of_features, 0);
    this->z = std::vector<float>(number_of_features, 0);
    this->z_delta = std::vector<float>(number_of_features, 0);
    this->delta_w = std::vector<float>(number_of_features, 0);

    this->h = std::vector<float>(number_of_features, 0);
    this->h_old = std::vector<float>(number_of_features, 0);
    this->h_temp = std::vector<float>(number_of_features, 0);
    this->beta = std::vector<float>(number_of_features, log(alpha_init));
    this->z_bar = std::vector<float>(number_of_features, 0);
    this->p = std::vector<float>(number_of_features, 0);

    this->v_old = 0;
    this->lambda = lambda_init;
    this->epsilon = epsilon_init;
    this->v_delta = 0;
    this->eta = eta_init;
    this->decay = decay_init;
    this->meta_step_size = meta_step_size_init;
}

float Math::DotProduct(const std::vector<float>& a, const std::vector<float>& b)
{
    float result = 0;
    for (int i = 0; i < a.size(); i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

float SwiftTDNonSparse::Step(const std::vector<float>& features, float reward)
{
    float v = 0;

    for (int i = 0; i < features.size(); i++)
    {
        v += this->w[i] * features[i];
    }

    float delta = reward + gamma * v - this->v_old;
    for (int i = 0; i < features.size(); i++)
    {
        this->delta_w[i] = delta * this->z[i] - z_delta[i] * this->v_delta;
        this->w[i] += this->delta_w[i];
        this->beta[i] +=
            this->meta_step_size / (exp(this->beta[i]) + 1e-8) * (delta - v_delta) * this->p[i];
        if (exp(this->beta[i]) > this->eta || isinf(exp(this->beta[i])))
        {
            this->beta[i] = log(this->eta);
        }
        this->h_old[i] = this->h[i];
        this->h[i] = this->h_temp[i] +
            delta * this->z_bar[i] - this->z_delta[i] * this->v_delta;
        this->h_temp[i] = this->h[i];
        z_delta[i] = 0;
        this->z[i] *= gamma * this->lambda;
        this->p[i] *= gamma * this->lambda;
        this->z_bar[i] *= gamma * this->lambda;
    }
    this->v_delta = 0;
    float tau = 0;
    for (int i = 0; i < features.size(); i++)
    {
        tau += exp(this->beta[i]) * features[i] * features[i];
    }
    float b = 0;
    for (int i = 0; i < features.size(); i++)
    {
        b += this->z[i] * features[i];
    }

    for (int i = 0; i < features.size(); i++)
    {
        this->v_delta += this->delta_w[i] * features[i];
        float multiplier = 1;
        if (eta / tau < 1)
        {
            multiplier = eta / tau;
        }
        this->z_delta[i] = multiplier * exp(this->beta[i]) * features[i];
        this->z[i] += this->z_delta[i] * (1 - b);
        this->p[i] += this->h_old[i] * features[i];
        this->z_bar[i] += this->z_delta[i] * (1 - b - this->z_bar[i] * features[i]);
        this->h_temp[i] = this->h[i] - this->h_old[i] * features[i] * (this->z[i] - this->z_delta[i]) -
            this->h[i] * this->z_delta[i] * features[i];
        if (tau > eta)
        {
            this->h_temp[i] = 0;
            this->h[i] = 0;
            this->h_old[i] = 0;
            this->z_bar[i] = 0;
            this->beta[i] += log(this->decay) * features[i] * features[i];
        }
    }
    this->v_old = v;
    return v;
}

SwiftTDBinaryFeatures::SwiftTDBinaryFeatures(int number_of_features, float lambda_init, float alpha_init,
                                             float gamma_init,
                                             float epsilon_init, float eta_init,
                                             float decay_init, float meta_step_size_init)
{
    this->gamma = gamma_init;
    this->w = std::vector<float>(number_of_features, 0);
    this->featureVector = std::vector<float>(number_of_features, 0);
    this->z = std::vector<float>(number_of_features, 0);
    this->z_delta = std::vector<float>(number_of_features, 0);
    this->delta_w = std::vector<float>(number_of_features, 0);

    this->h = std::vector<float>(number_of_features, 0);
    this->h_old = std::vector<float>(number_of_features, 0);
    this->h_temp = std::vector<float>(number_of_features, 0);
    this->beta = std::vector<float>(number_of_features, log(alpha_init));
    this->z_bar = std::vector<float>(number_of_features, 0);
    this->p = std::vector<float>(number_of_features, 0);

    this->last_alpha = std::vector<float>(number_of_features, 0);

    this->v_old = 0;
    this->lambda = lambda_init;
    this->epsilon = epsilon_init;
    this->v_delta = 0;
    this->eta = eta_init;
    this->decay = decay_init;

    this->meta_step_size = meta_step_size_init;
}

float SwiftTDBinaryFeatures::Step(const std::vector<int>& feature_indices, float reward)
{
    float v = 0;

    for (auto& index : feature_indices)
    {
        v += this->w[index];
    }

    float delta = reward + gamma * v - this->v_old;
    int position = 0;
    while (position < this->setOfEligibleItems.size())
    {
        int index = this->setOfEligibleItems[position];
        this->delta_w[index] = delta * this->z[index] - z_delta[index] * this->v_delta;
        this->w[index] += this->delta_w[index];
        this->beta[index] +=
            this->meta_step_size / (exp(this->beta[index]) + 1e-8) * (delta - v_delta) * this->p[index];
        if (exp(this->beta[index]) > this->eta || isinf(exp(this->beta[index])))
        {
            this->beta[index] = log(this->eta);
        }
        this->h_old[index] = this->h[index];
        this->h[index] = this->h_temp[index] +
            delta * this->z_bar[index] - this->z_delta[index] * this->v_delta;
        this->h_temp[index] = this->h[index];
        z_delta[index] = 0;
        this->z[index] = gamma * this->lambda * this->z[index];
        this->p[index] = gamma * this->lambda * this->p[index];
        this->z_bar[index] = gamma * this->lambda * this->z_bar[index];
        if (this->z[index] <= this->last_alpha[index] * epsilon)
        {
            this->z[index] = 0;
            this->p[index] = 0;
            this->z_bar[index] = 0;
            this->delta_w[index] = 0;
            this->setOfEligibleItems[position] = this->setOfEligibleItems[this->setOfEligibleItems.size() - 1];
            this->setOfEligibleItems.pop_back();
        }
        else
        {
            position++;
        }
    }
    this->v_delta = 0;
    float rate_of_learning = 0;

    for (auto& index : feature_indices)
    {
        rate_of_learning += exp(this->beta[index]);
    }
    float E = this->eta;
    if (rate_of_learning > this->eta)
    {
        E = rate_of_learning;
    }


    float t = 0;
    for (auto& index : feature_indices)
    {
        t += this->z[index];
    }

    for (auto& index : feature_indices)
    {
        if (z[index] == 0)
        {
            this->setOfEligibleItems.push_back(index);
        }
        this->v_delta += this->delta_w[index];
        this->z_delta[index] = (this->eta / E) * exp(this->beta[index]);
        this->last_alpha[index] = this->z_delta[index];
        if ((this->eta / E) < 1)
        {
            this->h_temp[index] = 0;
            this->h[index] = 0;
            this->h_old[index] = 0;
            this->z_bar[index] = 0;
            this->beta[index] += log(this->decay);
        }
        this->z[index] += this->z_delta[index] * (1 - t);
        this->p[index] += this->h_old[index];
        this->z_bar[index] += this->z_delta[index] * (1 - t - this->z_bar[index]);
        this->h_temp[index] = this->h[index] - this->h_old[index] * (this->z[index] - this->z_delta[index]) -
            this->h[index] * this->z_delta[index];
    }
    this->v_old = v;
    return v;
}


SwiftTD::SwiftTD(int number_of_features, float lambda_init, float alpha_init, float gamma_init,
                 float epsilon_init, float eta_init,
                 float decay_init, float meta_step_size_init)
{
    this->gamma = gamma_init;
    this->w = std::vector<float>(number_of_features, 0);
    this->featureVector = std::vector<float>(number_of_features, 0);
    this->z = std::vector<float>(number_of_features, 0);
    this->z_delta = std::vector<float>(number_of_features, 0);
    this->delta_w = std::vector<float>(number_of_features, 0);

    this->h = std::vector<float>(number_of_features, 0);
    this->h_old = std::vector<float>(number_of_features, 0);
    this->h_temp = std::vector<float>(number_of_features, 0);
    this->beta = std::vector<float>(number_of_features, log(alpha_init));
    this->z_bar = std::vector<float>(number_of_features, 0);
    this->p = std::vector<float>(number_of_features, 0);

    this->last_alpha = std::vector<float>(number_of_features, 0);

    this->v_old = 0;
    this->lambda = lambda_init;
    this->epsilon = epsilon_init;
    this->v_delta = 0;
    this->eta = eta_init;
    this->decay = decay_init;

    this->meta_step_size = meta_step_size_init;
}

float SwiftTD::Step(const std::vector<std::pair<int, float>>& feature_indices, float reward)
{
    float v = 0;

    for (auto& index : feature_indices)
    {
        v += this->w[index.first] * index.second;
    }

    float delta = reward + gamma * v - this->v_old;
    int position = 0;
    while (position < this->setOfEligibleItems.size())
    {
        auto index = this->setOfEligibleItems[position];
        this->delta_w[index.first] = delta * this->z[index.first] - z_delta[index.first] * this->v_delta;
        this->w[index.first] += this->delta_w[index.first];
        this->beta[index.first] +=
            this->meta_step_size / (exp(this->beta[index.first]) + 1e-8) * (delta - v_delta) * this->p[index.first];
        if (exp(this->beta[index.first]) > this->eta || isinf(exp(this->beta[index.first])))
        {
            this->beta[index.first] = log(this->eta);
        }
        this->h_old[index.first] = this->h[index.first];
        this->h[index.first] = this->h_temp[index.first] +
            delta * this->z_bar[index.first] - this->z_delta[index.first] * this->v_delta;
        this->h_temp[index.first] = this->h[index.first];
        z_delta[index.first] = 0;
        this->z[index.first] = gamma * this->lambda * this->z[index.first];
        this->p[index.first] = gamma * this->lambda * this->p[index.first];
        this->z_bar[index.first] = gamma * this->lambda * this->z_bar[index.first];
        if (this->z[index.first] <= this->last_alpha[index.first] * epsilon)
        {
            this->z[index.first] = 0;
            this->p[index.first] = 0;
            this->z_bar[index.first] = 0;
            this->delta_w[index.first] = 0;
            this->setOfEligibleItems[position] = this->setOfEligibleItems[this->setOfEligibleItems.size() - 1];
            this->setOfEligibleItems.pop_back();
        }
        else
        {
            position++;
        }
    }
    this->v_delta = 0;
    float rate_of_learning = 0;

    for (auto& index : feature_indices)
    {
        rate_of_learning += exp(this->beta[index.first]) * index.second * index.second;
    }
    float E = this->eta;
    if (rate_of_learning > this->eta)
    {
        E = rate_of_learning;
    }


    float t = 0;
    for (auto& index : feature_indices)
    {
        t += this->z[index.first] * index.second;
    }

    for (auto& index : feature_indices)
    {
        if (z[index.first] == 0)
        {
            this->setOfEligibleItems.push_back(index);
        }
        this->v_delta += this->delta_w[index.first] * index.second;
        this->z_delta[index.first] = (this->eta / E) * exp(this->beta[index.first]) * index.second;
        this->last_alpha[index.first] = this->z_delta[index.first];
        if ((this->eta / E) < 1)
        {
            this->h_temp[index.first] = 0;
            this->h[index.first] = 0;
            this->h_old[index.first] = 0;
            this->z_bar[index.first] = 0;
            this->beta[index.first] += log(this->decay) * index.second * index.second;
        }
        this->z[index.first] += this->z_delta[index.first] * (1 - t);
        this->p[index.first] += this->h_old[index.first] * index.second;
        this->z_bar[index.first] += this->z_delta[index.first] * (1 - t - this->z_bar[index.first] * index.second);
        this->h_temp[index.first] = this->h[index.first] - this->h_old[index.first] * index.second * (this->z[index.
                    first] - this->
                z_delta[index.first]) -
            this->h[index.first] * this->z_delta[index.first] * index.second;
    }
    this->v_old = v;
    return v;
}
