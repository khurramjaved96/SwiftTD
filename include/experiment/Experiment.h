//
// Created by Khurram Javed on 2021-03-14.
//

#ifndef INCLUDE_EXPERIMENT_EXPERIMENT_H_

#define INCLUDE_EXPERIMENT_EXPERIMENT_H_

#include <string>
#include <vector>
#include <map>
#include "Database.h"
#include "../json.hpp"

class Experiment {
 protected:
  std::map<std::string, std::vector<std::string>> args;

  std::string output_dir;
  Database d = Database();

  static std::vector<int> frequency_of_params(std::map<std::string, std::vector<std::string>> &args);

 public:
  std::map<std::string, std::string> args_for_run;
  int run;
  std::string database_name;

  Experiment(int name, char *argv[]);
  Experiment();
  static std::map<std::string, std::vector<std::string>> parse_params(int total_prams, char *pram_list[]);

  int get_int_param(const std::string &);

  float get_float_param(const std::string &);

  std::string get_string_param(const std::string &);

  std::vector<std::string> get_vector_param(const std::string&);
};

class ExperimentJSON : public Experiment {
 protected:
  int get_prod(std::vector<int>);
 public:
  ExperimentJSON();
  std::vector<std::pair<std::string, std::string>> get_all_keys(nlohmann::json j);
  ExperimentJSON(int argc, char *argv[]);
  std::vector<int> get_combinations(nlohmann::json j);
  std::map<std::string, std::string> get_args_for_run(nlohmann::json j, int rank);
};

class CountConfig : public ExperimentJSON{
 public:
  CountConfig(int argc, char *argv[]);
};

#endif  // INCLUDE_EXPERIMENT_EXPERIMENT_H_
