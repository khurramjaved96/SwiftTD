//
// Created by Khurram Javed on 2022-07-19.
//

#ifndef INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_
#define INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_

#include "../../src/src/ale_interface.hpp"
#include "../../src/src/common/Constants.h"
#include <string>
#include <vector>

class ProtoPredictionEnvironment {
protected:
  std::vector<unsigned char> gray_features;
  std::vector<float> observation;
  bool to_reset;
  std::vector<float> list_of_rewards;
  std::vector<float> list_of_returns;
  ale::ALEInterface my_env;
  std::vector<ale::Action> action_set;
  char *actions;
  float reward;
  float ep_reward;
  void UpdateReturns();

public:
  std::vector<float> real_target;
  int time;
  int total;
  float gamma;
  ProtoPredictionEnvironment(std::string path, float gamma);
  std::vector<float>& GetListOfReturns();
  std::vector<float> get_state();
  std::vector<float> step();
  std::vector<float> FastStep();
  float get_target();
  float get_gamma();
  bool get_done();
  float get_reward();
};

//class AtariLarge : public ProtoPredictionEnvironment{
//public:
//  AtariLarge(std::string path, float gamma);
//  std::vector<float> step() override;
//};

#endif // INCLUDE_ENVIRONMENTS_PROTO_PREDICTION_ENVIRONMENTS_H_
