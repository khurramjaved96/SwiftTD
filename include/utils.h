//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>


template <class T>
void print_vector(std::vector<T> const &v){
  std::cout << "[";
  int counter = 0; for (auto i = v.begin(); i != v.end(); ++i) {
    std::cout << " " << std::setw(3) << *i << ",";
    if (counter % 100 == 99) std::cout << "\n";
      counter++;
  }
  std::cout << "]\n";
}

void print_vector(std::vector < int >
const &v);

void print_vector(std::vector < float >
const &v);

void print_vector(std::vector < char >
const &v);


void print_vector(std::vector < long unsigned int >
const &v);

void print_matrix(std::vector < std::vector < int >>
const &v);

void print_matrix(std::vector < std::vector < float >>
const &v);

//class NetworkVisualizer {
//  std::string dot_string;
//  std::vector<NeuronSy *> all_neurons;
//
// public:
//  explicit NetworkVisualizer(std::vector<Neuron *> all_neurons);
//
//  void generate_dot(int time_step);
//
//  std::string get_graph(int time_step);
//
//  std::string get_graph_detailed(int time_step);
//
//  void generate_dot_detailed(int time_step);
//};

#endif  // INCLUDE_UTILS_H_

