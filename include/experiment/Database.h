//
// Created by Khurram Javed on 2021-03-29.
//

#include <mysql.h>
#include <string>
#include <iostream>
#include <vector>

#ifndef INCLUDE_EXPERIMENT_DATABASE_H_
#define INCLUDE_EXPERIMENT_DATABASE_H_

class Database {
  MYSQL *mysql;

  static std::string vec_to_tuple(std::vector<std::string> row, const std::string &padding);

  int connect();

  int connect_and_use(std::string database_name);

  int run_query(std::string query, const std::string &database_name);

 public:

  int create_database(const std::string &database_name);

  int
  add_rows_to_table(const std::string &database_name, const std::string &table, const std::vector<std::string> &keys,
                    const std::vector<std::vector<std::string>> &values);

  int add_row_to_table(const std::string &database_name, const std::string &table, std::vector<std::string> keys,
                       std::vector<std::string> values);



  int make_table(const std::string &database_name, const std::string &table, std::vector<std::string> keys,
                 std::vector<std::string> types,
                 std::vector<std::string> index_columns);

  Database();
};

#endif  // INCLUDE_EXPERIMENT_DATABASE_H_
