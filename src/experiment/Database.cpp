//
// Created by Khurram Javed on 2021-03-29.
//

#include "../../include/experiment/Database.h"
#include <mysql.h>
#include <string>
#include <iostream>
#include <cstring>
#include <stdlib.h>

#include <chrono>
#include <thread>
#include <random>

using namespace std::chrono_literals;

Database::Database() {
  this->mysql = mysql_init(NULL);

  if (!this->mysql) {
    puts("Init faild, out of memory?");
  }

  mysql_options(this->mysql, MYSQL_READ_DEFAULT_FILE, (void *) ".my.cnf");
}

/// Connets to the database and stores the connection in this->mysql
/// \return 0 if successfull, non-zero otherwise
int Database::connect() {
  this->mysql = mysql_init(NULL);

  if (!this->mysql) {
    puts("Init faild, out of memory?");
  }
//
  mysql_options(this->mysql, MYSQL_READ_DEFAULT_FILE, (void *) ".my.cnf");
  mysql_real_connect(this->mysql,       /* MYSQL structure to use */
                     NULL,  /* server hostname or IP address */
                     NULL,  /* mysql user */
                     NULL,   /* password */
                     NULL,        /* default database to use, NULL for none */
                     0,           /* port number, 0 for default */
                     NULL,        /* socket file or named pipe name */
                     CLIENT_FOUND_ROWS /* connection flags */);
  return 0;
}

/// Connect to the database and execute USE this->db_name;
/// \return
int Database::connect_and_use(std::string database_name) {
  this->mysql = mysql_init(NULL);

  if (!this->mysql) {
    puts("Init faild, out of memory?");
  }
//
  mysql_options(this->mysql, MYSQL_READ_DEFAULT_FILE, (void *) ".my.cnf");
  mysql_real_connect(this->mysql,       /* MYSQL structure to use */
                     NULL,  /* server hostname or IP address */
                     NULL,  /* mysql user */
                     NULL,   /* password */
                     NULL,        /* default database to use, NULL for none */
                     0,           /* port number, 0 for default */
                     NULL,        /* socket file or named pipe name */
                     CLIENT_FOUND_ROWS /* connection flags */);
  std::string use_query = "USE " + database_name + ";";
//    std::cout << "Running query " << use_query << std::endl;
  int selection = mysql_query(this->mysql, &use_query[0]);
  return 0;
}

int Database::create_database(const std::string &database_name) {
  std::string query = "CREATE DATABASE " + database_name + ";";
  std::cout << query << std::endl;
  this->connect();
  int val = mysql_query(this->mysql, &query[0]);
  if (val) {
    std::cout << "Database creation failed. Perhaps it already exists\n";
  }
  mysql_commit(this->mysql);
  mysql_close(this->mysql);
  std::cout << "Database created\n";
  return val;
}

int Database::run_query(std::string query, const std::string &database_name) {
  this->connect_and_use(database_name);
  mysql_query(this->mysql, &query[0]);
  return 1;
}

std::string Database::vec_to_tuple(std::vector<std::string> row, const std::string &padding) {
  std::string tup = "(";
  for (int counter = 0; counter < row.size() - 1; counter++) {

    if(row[counter] == "-nan" || row[counter] == "nan" ||  row[counter] == "inf" ||  row[counter] == "-inf")
      tup += "NULL";
    else {
      tup += padding;
      tup += row[counter];
      tup += padding;
    }
    tup += ",";
  }
  if(row[row.size() - 1] == "-nan" || row[row.size() - 1] == "nan" || row[row.size() - 1] == "inf" || row[row.size() - 1] == "-inf" )
    tup = tup  + "NULL" + " )";
  else
    tup = tup + padding + row[row.size() - 1] + padding + " )";
  return tup;
}

//int Database::add_rows_to_table(const std::string &database_name, const std::string &table,
//                                const std::vector<std::string> &keys,
//                                const std::vector<std::vector<std::string>> &values) {
//  this->connect_and_use(database_name);
//  mysql_autocommit(this->mysql, 0);
//  for (auto &value : values) {
//    std::string query = "INSERT INTO " + table + vec_to_tuple(keys, "") + " VALUES " + vec_to_tuple(value, "'");
//    mysql_query(this->mysql, &query[0]);
//  }
//  mysql_commit(this->mysql);
//  mysql_close(this->mysql);
//  return 0;
//}

// Much faster
int Database::add_rows_to_table(const std::string &database_name, const std::string &table,
                                const std::vector<std::string> &keys,
                                const std::vector<std::vector<std::string>> &values) {
  float return_val = 1;
  std::string query = "INSERT INTO " + table + vec_to_tuple(keys, "") + " VALUES ";
  for (auto &value : values) {
    query += vec_to_tuple(value, "'");
    if (&value != &values.back())
      query += ",";
  }
  int failures = 0;
  using namespace std::this_thread;     // sleep_for, sleep_until
  using namespace std::chrono_literals; // ns, us, ms, s, h, etc.

  using std::chrono::system_clock;
  std::mt19937 mt;
  std::uniform_int_distribution<int> time_sampler(50, 3000);
  while (return_val && failures < 10) {
    this->connect_and_use(database_name);
    return_val = mysql_query(this->mysql, &query[0]);
    if(return_val == 0 || return_val == 1) // it returns 1 on success for me ¯\_(ツ)_/¯
      return_val = mysql_commit(this->mysql);
    if(return_val != 0 && return_val != 1){
      std::cout << "Error code = " << return_val << std::endl;
      int sleep_time = time_sampler(mt);
      std::cout << "Attempt " << failures << " failed;"  <<  " Sleeping for " << sleep_time << " ms" << std::endl;
      sleep_for(std::chrono::milliseconds(sleep_time));
      failures++;
//      std::cout << "Query commit failed\n";
//      std::cout << query << std::endl;
    }
    mysql_close(this->mysql);
  }
  return 0;
}

int
Database::add_row_to_table(const std::string &database_name, const std::string &table, std::vector<std::string> keys,
                           std::vector<std::string> values) {
  std::string query = "INSERT INTO " + table + vec_to_tuple(keys, "") + " VALUES " + vec_to_tuple(values, "'");
  this->run_query(query, database_name);
  mysql_commit(this->mysql);
  mysql_close(this->mysql);
  return 0;
}

int Database::make_table(const std::string &database_name, const std::string &table, std::vector<std::string> keys,
                         std::vector<std::string> types,
                         std::vector<std::string> index_columns) {

  std::string query;
  query = "CREATE TABLE " + table + " (";
  if (keys.size() != types.size()) {
    std::cout << "SQL number of columns and number of types are not equal\n";
    exit(1);
  }
  for (int counter = 0; counter < keys.size(); counter++) {
    query += " " + keys[counter] + " " + types[counter] + ",";
  }
  query = query + " PRIMARY KEY(";
  for (int counter = 0; counter < index_columns.size() - 1; counter++) {
    query += " " + index_columns[counter] + " ,";
  }
  query = query + " " + index_columns[index_columns.size() - 1] + " ));";
  std::cout << "Creating table: " << query << std::endl;
  this->run_query(query, database_name);
  mysql_close(this->mysql);

  return 1;
}

