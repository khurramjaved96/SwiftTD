//
// Created by Khurram Javed on 2021-03-14.
//

#include "../../include/experiment/Experiment.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include "../../include/experiment/Database.h"
#include "../../include/json.hpp"
#include <fstream>
#include "../../include/utils.h"

std::map<std::string, std::vector<std::string>> Experiment::parse_params(int total_prams, char **pram_list) {
  bool argument = false;
  std::map<std::string, std::vector<std::string>> result;
  std::string key;
  for (int temp = 0; temp < total_prams; temp++) {
    std::string s(pram_list[temp]);
    if (s.find("--") != std::string::npos) {
//            -- available os a new parameter
      argument = true;
      key = pram_list[temp];
      key = key.substr(2, key.size() - 2);
      std::vector<std::string> my_vec;
      result.insert(std::pair<std::string, std::vector<std::string >>(key, my_vec));
    } else if (argument) {
      std::string value = pram_list[temp];
      result[key].push_back(value);
    }
  }
  return result;
}

std::vector<std::string> Experiment::get_vector_param(const std::string &key) {
  std::string val = this->get_string_param(key);
  std::vector<std::string> return_vec;
  int initial_index = 0;
  int final_index = -1;
  std::string temp_str;
  for (int i = 0; i < val.size(); i++) {
    if (val[i] == ':' || i == val.size() - 1) {
      final_index = i;
      if (i == val.size() - 1)
        temp_str.append(val, initial_index, final_index + 1 - initial_index);
      else
        temp_str.append(val, initial_index, final_index - initial_index);
      return_vec.push_back(temp_str);
      initial_index = final_index + 1;
      temp_str.clear();

    }
  }
//  for (int i = 0; i < return_vec.size(); i++)
//    std::cout << "Arg = " << return_vec[i] << std::endl;
  return return_vec;
}

std::vector<int> Experiment::frequency_of_params(std::map<std::string, std::vector<std::string>> &args) {
  int total_combinations = 1;
  std::vector<int> size_of_params;
  for (auto it = args.begin();
       it != args.end(); it++) {
    if (it->first == "run") {
      assert(it->second.size() == 1);
    }
    size_of_params.push_back(it->second.size());
    total_combinations *= it->second.size();
  }
  return size_of_params;
}

Experiment::Experiment() {}

int ExperimentJSON::get_prod(std::vector<int> my_vec) {
  int prod = 1;
  for (int i = 0; i < my_vec.size(); i++)
    prod *= my_vec[i];
  return prod;
}
std::vector<int> ExperimentJSON::get_combinations(nlohmann::json j) {
  using namespace nlohmann;

  std::vector<int> combination_list;
  int total_combinations = 1;
  json::iterator it = j.begin();
  while (it != j.end()) {
    if (it->is_array()) {
      int object_combi = 0;
      for (int inner = 0; inner < it->size(); inner++) {
        if ((*it)[inner].is_object()) {
          object_combi += get_prod(get_combinations((*it)[inner]));
        } else if ((*it)[inner].is_primitive()) {
          object_combi++;
        }
      }
      total_combinations *= object_combi;
      combination_list.push_back(object_combi);
    } else if (it->is_object()) {
      int temp_comb = get_prod(this->get_combinations(*it));
      total_combinations *= temp_comb;
      combination_list.push_back(temp_comb);
    } else if (it->is_primitive()) {
      combination_list.push_back(1);
    }
    it++;
  }

  return combination_list;
}

std::map<std::string, std::string> ExperimentJSON::get_args_for_run(nlohmann::json j, int rank) {
  std::map<std::string, std::string> my_map;
  std::vector<int> freq = this->get_combinations(j);
//  std::cout << "Freq = " << std::endl;
//  print_vector(freq);
  int total_combinations = get_prod(freq);
  rank = rank % total_combinations;
  int counter = 0;
//  std::cout << j << std::endl;
  for (auto it = j.begin(); it != j.end(); ++it) {
//    std::cout << "Counter " << counter << std::endl;
//    std::cout << *it << std::endl;
//    print_vector(freq);
//    std::cout << freq[counter] << " " << rank << std::endl;;
    int temp_index = rank % freq[counter];
//    std::cout << "mod done\n";
    rank /= freq[counter];
//    std::cout << "Val = " << (*it) << std::endl;
//    std::cout << "Is array " << it->is_array() << std::endl;
//    std::cout << "Is prim " << it->is_primitive() << std::endl;
//    std::cout << "Is obj " << it->is_object() << std::endl;
    if (it->is_primitive()) {
//      std::cout << "Is prim\n";
      my_map.insert(std::pair<std::string, std::string>(it.key(), *it));
    } else if (it->is_array() && (*it)[0].is_primitive()) {
//      std::cout << "Is array of prim\n";
//      std::cout << "Primitive iterator " << std::endl;
//      std::cout << "Temp index = " << temp_index << std::endl;
      my_map.insert(std::pair<std::string, std::string>(it.key(), (*it)[temp_index]));
//      std::cout << "Key " << it.key() << std::endl;
//      std::cout << "Value" << it[temp_index] << std::endl;
    } else if (!it->is_array() && it->is_object()) {
//      std::cout << "Is obj\n";
//      std::cout << "Calling recursive on " << *it << std::endl;
      auto temp_map = this->get_args_for_run(*it, temp_index);
      my_map.insert(temp_map.begin(), temp_map.end());
    } else if (it->is_array() && (*it)[0].is_object()) {
//      std::cout << "Is array of obj\n";
//      std::cout << "Calling recursive on 2" << *it << std::endl;
      auto freq = this->get_combinations(*it);
      int inner_index = 0;
      int sum = freq[0];
      int old_sum = 0;
      while (temp_index >= sum) {
        inner_index++;
        old_sum = sum;
        sum += freq[inner_index];
      }
      temp_index = temp_index - old_sum;
      auto temp_j_j = (*it)[inner_index];
      for (auto it2 = temp_j_j.begin(); it2 != temp_j_j.end(); ++it2) {
//        std::cout << it.key() << " " << it2.key() << std::endl;
        my_map.insert(std::pair<std::string, std::string>(it.key(), it2.key()));
      }
//      std::cout << "IMPORTANT VALUE IMPORTANT VMAF\n" << it.key() << " val val " << (*it)[inner_index].back() << std::endl;
//      my_map.insert(std::pair<std::string, std::string>(it.key(), (*it)[inner_index]));
      auto temp_map = this->get_args_for_run((*it)[inner_index], temp_index);
      my_map.insert(temp_map.begin(), temp_map.end());
    }
    counter++;
  }
  return my_map;
}

std::vector<std::pair<std::string, std::string>> ExperimentJSON::get_all_keys(nlohmann::json j) {
  std::vector<std::pair<std::string, std::string>> key_values;
//  std::vector<std::string> values;
  for (auto it = j.begin(); it != j.end(); ++it) {
//    std::cout << *it << std::endl;
//    if(it->is_primitive() || it->is_array()) {
//      keys.push_back(it.key());
    if (it->is_primitive()) {
      key_values.emplace_back(it.key(), *it);
//        values.push_back(*it);
    } else if (it->is_array() && (*it)[0].is_object()) {
//        auto temp_j = (*it)[0];
//        auto j_2 = temp_j.begin();
      key_values.emplace_back(it.key(), (*it)[0].begin().key());
//        values.push_back((*it)[0]);
    } else if (it->is_array() && !(*it)[0].is_object()) {
      key_values.emplace_back(it.key(), (*it)[0]);
    }
//    }

    if (it->is_object() || (it->is_array() && (*it)[0].is_object())) {
      auto keys_temp = get_all_keys(*it);
      for (const auto &temp : keys_temp)
        key_values.push_back(temp);
    }
  }
  return key_values;
}

ExperimentJSON::ExperimentJSON() {}

CountConfig::CountConfig(int argc, char **argv) {
  using namespace nlohmann;
  this->args = this->parse_params(argc, argv);
  std::string file_name;
  if (this->args.count("config"))
    file_name = this->args["config"][0];
  else {
    std::cout << "Config file not provided; quiting\n";
    exit(0);
  }

  std::ifstream myfile;
  myfile.open(file_name);
  nlohmann::json j;
  myfile >> j;
  int total_combinations = get_prod(get_combinations(j));
  std::cout << total_combinations << std::endl;
}

ExperimentJSON::ExperimentJSON(int argc, char **argv) {
  using namespace nlohmann;
  this->args = this->parse_params(argc, argv);
  std::string file_name;
  if (this->args.count("config"))
    file_name = this->args["config"][0];
  else {
    std::cout << "Config file not provided; quiting\n";
    exit(0);
  }
  int run;
  if (this->args.count("run"))
    run = std::stoi(this->args["run"][0]);
  else {
    std::cout << "Run ID not provided" << std::endl;
    exit(1);
  }

  std::ifstream myfile;
  myfile.open(file_name);
  nlohmann::json j;
  myfile >> j;
  int total_combinations = get_prod(this->get_combinations(j));
  std::cout << "TOTAL COMBINATIONS = " << total_combinations << std::endl;
  json::iterator it = j["experiment"]["params"].begin();
  args_for_run = this->get_args_for_run(j, run);
  args_for_run.insert(std::pair<std::string, std::string>("run", std::to_string(run)));
//  std::cout << "PRINTING MAP PRINTING MAP PRINTING MAP\n";
  for (auto it = args_for_run.begin();
       it != args_for_run.end(); ++it) {
    std::cout << it->first << " " << it->second << "\n";
  }
  auto all_keys = this->get_all_keys(j);
  std::cout << "All keys \n";
  std::vector<std::string> keys, values, types;

  for (auto const &imap: all_keys) {
    keys.push_back(imap.first);
    if (!imap.second.empty() && imap.second.find_first_not_of("-0123456789") == std::string::npos) {
      types.emplace_back("int");
    } else if (!imap.second.empty() && imap.second.find_first_not_of("-.0123456789") == std::string::npos) {
      types.emplace_back("real");
    } else
      types.emplace_back("text");
  }
  keys.push_back("run");
  types.push_back("int");

  this->database_name = "khurram_" + this->args_for_run["name"];
  this->d.create_database(this->database_name);
  this->d.make_table(this->database_name, "runs", keys, types, std::vector<std::string>{"run"});
  keys.clear();
  values.clear();
  types.clear();

  for (auto const &imap: this->args_for_run) {
    keys.push_back(imap.first);
    values.push_back(imap.second);
  }

  this->d.add_row_to_table(this->database_name, "runs", keys, values);
  return;
//  this->args = this->parse_params(argc, argv);
//
//  std::vector<int> size_of_params = this->frequency_of_params(this->args);
//  if (this->args.count("run")) {
//    this->run = std::stoi(this->args["run"][0]);
//  } else {
//    std::cout << "Run number not provided; for example, pass --run 0 as command line argument. Exiting \n";
//    exit(0);
//  }
//
//  int temp_rank = this->run;
//
//  std::vector<int> selected_combinations;
//  for (int &size_of_param : size_of_params) {
//    selected_combinations.push_back(temp_rank % size_of_param);
//    temp_rank = temp_rank / size_of_param;
//  }
//  int temp_counter = 0;
//  for (auto &arg : this->args) {
//    this->args_for_run.insert(
//        std::pair<std::string, std::string>(arg.first, arg.second[selected_combinations[temp_counter]]));
//    std::cout << arg.first << " " << arg.second[selected_combinations[temp_counter]] << std::endl;
//    temp_counter++;
//  }
//
//  this->database_name = "khurram_" + this->args_for_run["name"];
//  this->d.create_database(this->database_name);
//  std::vector<std::string> keys, values, types;
//  for (auto const &imap: this->args_for_run) {
//    keys.push_back(imap.first);
//    if (!imap.second.empty() && imap.second.find_first_not_of("-0123456789") == std::string::npos) {
//      types.emplace_back("int");
//    } else if (!imap.second.empty() && imap.second.find_first_not_of("-.0123456789") == std::string::npos) {
//      types.emplace_back("real");
//    } else
//      types.emplace_back("text");
//
//    values.push_back(imap.second);
//
//  }
//  this->d.make_table(this->database_name, "runs", keys, types, std::vector<std::string>{"run"});
//  this->d.add_row_to_table(this->database_name, "runs", keys, values);
}
Experiment::Experiment(int argc, char *argv[]) {

  this->args = this->parse_params(argc, argv);
  std::vector<int> size_of_params = this->frequency_of_params(this->args);
  if (this->args.count("run")) {
    this->run = std::stoi(this->args["run"][0]);
  } else {
    std::cout << "Run number not provided; for example, pass --run 0 as command line argument. Exiting \n";
    exit(0);
  }

  int temp_rank = this->run;

  std::vector<int> selected_combinations;
  for (int &size_of_param : size_of_params) {
    selected_combinations.push_back(temp_rank % size_of_param);
    temp_rank = temp_rank / size_of_param;
  }
  int temp_counter = 0;
  for (auto &arg : this->args) {
    this->args_for_run.insert(
        std::pair<std::string, std::string>(arg.first, arg.second[selected_combinations[temp_counter]]));
    std::cout << arg.first << " " << arg.second[selected_combinations[temp_counter]] << std::endl;
    temp_counter++;
  }

  this->database_name = "khurram_" + this->args_for_run["name"];
  this->d.create_database(this->database_name);
  std::vector<std::string> keys, values, types;
  for (auto const &imap: this->args_for_run) {
    keys.push_back(imap.first);
    if (!imap.second.empty() && imap.second.find_first_not_of("-0123456789") == std::string::npos) {
      types.emplace_back("int");
    } else if (!imap.second.empty() && imap.second.find_first_not_of("-.0123456789") == std::string::npos) {
      types.emplace_back("real");
    } else
      types.emplace_back("text");

    values.push_back(imap.second);

  }
  this->d.make_table(this->database_name, "runs", keys, types, std::vector<std::string>{"run"});
  this->d.add_row_to_table(this->database_name, "runs", keys, values);
}

int Experiment::get_int_param(const std::string &param) {
//    std::cout << "Param count " << param << this->args_for_run.count(param) << " " << std::endl;
  if (this->args_for_run.count(param) == 0) {
    std::cout << "Param does not exist\n";
    throw std::invalid_argument("Param " + param + " does not exist");
  }
  return std::stoi(this->args_for_run[param]);
}

float Experiment::get_float_param(const std::string &param) {
//    std::cout << "Param count " << param << this->args_for_run.count(param) << " " << std::endl;
  if (this->args_for_run.count(param) == 0) {
    std::cout << "Param does not exist\n";
    throw std::invalid_argument("Param " + param + " does not exist");
  }
  return std::stof(this->args_for_run[param]);
}

std::string Experiment::get_string_param(const std::string &param) {
//    std::cout << "Param count " << param << this->args_for_run.count(param) << " " << std::endl;
  if (this->args_for_run.count(param) == 0) {
    std::cout << "Param does not exist\n";
    throw std::invalid_argument("Param " + param + " does not exist");
  }
  return this->args_for_run[param];
}