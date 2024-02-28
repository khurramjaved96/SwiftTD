//
// Created by Khurram Javed on 2022-07-19.
//

#include "../../include/environments/proto_prediction_environments.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

ProtoPredictionEnvironment::ProtoPredictionEnvironment(std::string game_name,
                                                       float gamma)
        : gray_features(210 * 160, 0), observation(84 * 84 * 16 + 20, 0) {

    this->gamma = gamma;
    this->gamma = gamma;
    std::fstream policy("../policies/" + game_name + "NoFrameskip-v4.txt",
                        std::ios::in | std::ios::binary);
    my_env.setInt("random_seed", 1731038949);
    //  my_env.setBool("truncate_on_loss_of_life", true);
    my_env.setFloat("repeat_action_probability", 0.0);
    my_env.setInt("frame_skip", 1);
    my_env.loadROM("../games/" + game_name + ".bin");
    my_env.reset_game();

    long size;
    policy.seekg(0, std::ios::end);
    size = policy.tellg();
    policy.seekg(0, std::ios::beg);
    actions = new char[size];
    policy.read(actions, size);
    policy.close();
    std::cout << "Size of actions = " << size << std::endl;
    action_set = my_env.getMinimalActionSet();
    time = 0;
    reward = 0;
    ep_reward = 0;
    to_reset = false;
}


std::vector<float> ProtoPredictionEnvironment::get_state() {
    //make a vector the size of this->observation
    std::vector<float> state(this->observation.size(), 0);
    // loop over observation. If value is above 1, set it to one in state
    for (int i = 0; i < this->observation.size(); i++) {
        if (this->observation[i] > 0.1) {
            state[i] = 1;
        }
    }
    return state;
}

void ProtoPredictionEnvironment::UpdateReturns() {
    float old_val = 0;
    list_of_returns = std::vector<float>(list_of_rewards.size(), 0);
    for (int i = list_of_rewards.size() - 1; i >= 0; i--) {
        list_of_returns[i] = list_of_rewards[i] + old_val;
        old_val = list_of_returns[i] * this->gamma;
    }
    list_of_rewards.clear();
}

// S, 1, S, 1, S, R,
std::vector<float> &ProtoPredictionEnvironment::GetListOfReturns() {
    return this->list_of_returns;
}

bool ProtoPredictionEnvironment::get_done() { return true; }

std::vector<float> FastStep(){

}
std::vector<float> ProtoPredictionEnvironment::step() {
    to_reset = false;
    time++;
    for (int i = 84 * 84 * 16; i < 84 * 84 * 16 + 20; i++) {
        observation[i] = 0;
    }
    if (actions[time] == 'R') {
        reward = 0;
        my_env.reset_game();
        to_reset = true;
    } else {
        reward = my_env.act(action_set[int(actions[time]) - 97]);
        observation[int(actions[time]) - 97 + 84 * 84 * 16] = 1;
        observation[84 * 84 * 16 + 19] = get_reward();
    }
    my_env.getScreenGrayscale(gray_features);
    cv::Mat image(210, 160, CV_8UC1, gray_features.data());
    cv::Mat dest;
    cv::resize(image, dest, cv::Size(84, 84));
    // decay all values of observation by 0.9
    for (int i = 0; i < 84 * 84 * 16; i++) {
        observation[i] -=1;
    }
    // Set new values to 1
    for (int i = 0; i < 84 * 84; i++) {
        observation[i * 16 + int(dest.data[i] / 16)] = 8;
    }


    return this->get_state();
}



//std::vector<float> AtariLarge::step() {
//  return observation;
//}


std::vector<float> ProtoPredictionEnvironment::FastStep(){
    to_reset = false;
    time++;

    if (actions[time] == 'R') {
        my_env.reset_game();
        to_reset = true;
    } else {
        reward = my_env.act(action_set[int(actions[time]) - 97]);
    }

    return {};
}

float ProtoPredictionEnvironment::get_target() { return real_target[time]; }

float ProtoPredictionEnvironment::get_gamma() {
    if (to_reset)
        return 0;
    return gamma;
}

float ProtoPredictionEnvironment::get_reward() {
    if (reward > 0.3)
        return 1;
    else if (reward < -0.3)
        return -1;

    return 0;
}
