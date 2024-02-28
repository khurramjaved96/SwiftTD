//
// Created by Khurram Javed on 2021-03-30.
//

#include <utility>
#include <vector>
#include "../../include/experiment/Database.h"
#include "../../include/experiment/Metric.h"

Metric::Metric(std::string db_name, std::string table_name, std::vector<std::string> keys,
               std::vector<std::string> type,
               std::vector<std::string> index) {
  this->database_name = db_name;
  this->table_name = std::move(table_name);
  this->db_columns = std::move(keys);
  this->db_types = std::move(type);
  this->index_columns = std::move(index);
  this->d.make_table(this->database_name, this->table_name, this->db_columns, this->db_types, this->index_columns);
}

void Metric::record_value(std::vector<std::string> values) {
  this->data_vector.push_back(values);
}

void Metric::commit_values() {
  if(this->data_vector.size() > 0) {
    this->add_values(this->data_vector);
    this->data_vector.clear();
  }
}

int Metric::add_value(std::vector<std::string> values) {
  return this->d.add_row_to_table(this->database_name, this->table_name, this->db_columns, values);

}

int Metric::add_values(const std::vector<std::vector<std::string>> &vector_of_values) {
  this->d.add_rows_to_table(this->database_name, this->table_name, this->db_columns, vector_of_values);
  return 0;
}

