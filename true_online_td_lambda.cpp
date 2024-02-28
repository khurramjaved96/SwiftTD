//
// Created by Khurram Javed on 2024-02-18.
//
//#define GUI_ENABLED

#include <iomanip>
#include <iostream>

#include <vector>
#include <fstream>
#include "include/neural_network/LinearLearner.h"
#include "include/environments/proto_prediction_environments.h"
#include <iostream>
#include <random>
#include "include/experiment/Metric.h"
#include "include/experiment/Experiment.h"
#include "src/src/ale_interface.hpp"

Experiment *my_experiment;

int main(int argc, char *argv[]) {

    my_experiment = new ExperimentJSON(argc, argv);


    Metric predictions = Metric(my_experiment->database_name, "predictions",
                                std::vector<std::string>{"run", "step", "prediction", "reward", "return_val"},
                                std::vector<std::string>{"int", "int", "real", "real", "real"},
                                std::vector<std::string>{"run", "step"});
    Metric lifetime_error_metric = Metric(my_experiment->database_name, "error",
                                          std::vector<std::string>{"run", "step", "mse"},
                                          std::vector<std::string>{"int", "int", "real"},
                                          std::vector<std::string>{"run", "step"});

    Metric step_sizes = Metric(my_experiment->database_name, "stepsize",
                               std::vector<std::string>{"run", "elem", "stepsize"},
                               std::vector<std::string>{"int", "int", "real"},
                               std::vector<std::string>{"run", "elem"});


    ProtoPredictionEnvironment env2(my_experiment->get_string_param("env"),
                                    my_experiment->get_float_param("gamma"));


    std::cout << "Starting experiment\n";
    LinearLearner *t;
    if (my_experiment->get_string_param("learner") == "SwiftTD") {
        t = new SwiftTD(my_experiment->get_int_param("features"),
                        my_experiment->get_float_param("lambda"),
                        my_experiment->get_float_param("initial_step_size"),
                        my_experiment->get_float_param("gamma"),
                        my_experiment->get_float_param("eps"),
                        my_experiment->get_float_param("max_step_size"),
                        my_experiment->get_float_param("step_size_decay"),
                        my_experiment->get_float_param("meta_step_size"),
                        my_experiment->get_float_param("weight_decay"));
    } else if (my_experiment->get_string_param("learner") == "semi_gradient_td_lambda") {
        t = new SemiGradientTDLambda(my_experiment->get_int_param("features"),
                                     my_experiment->get_float_param("lambda"),
                                     my_experiment->get_float_param("meta_step_size"),
                                     my_experiment->get_float_param("initial_step_size"),
                                     my_experiment->get_float_param("gamma"),
                                     my_experiment->get_float_param("eps"));
    } else if (my_experiment->get_string_param("learner") == "full_gradient_td_lambda") {
        t = new FullGradientTDLambda(my_experiment->get_int_param("features"),
                                     my_experiment->get_float_param("lambda"),
                                     my_experiment->get_float_param("meta_step_size"),
                                     my_experiment->get_float_param("initial_step_size"),
                                     my_experiment->get_float_param("gamma"),
                                     my_experiment->get_float_param("eps"));
    } else if (my_experiment->get_string_param("learner") == "true_online_td_lambda") {
        t = new TrueOnlineTDLambda(my_experiment->get_int_param("features"),
                                   my_experiment->get_float_param("lambda"),
                                   my_experiment->get_float_param("initial_step_size"),
                                   my_experiment->get_float_param("gamma"));
    } else {
        std::cout << "Invalid learner\n";
        return 0;
    }


    std::vector<float> list_of_predictions;
    std::vector<float> list_of_rewards;
    std::vector<float> steps;
    auto x = env2.step();
    float old_pred = t->Step(x, 0);
//        list_of_predictions.push_back(old_pred);


    float running_error = 0;
    for (int i = 0; i < my_experiment->get_int_param("seed"); i++) {
        env2.FastStep();
    }
    float error = 0;
    float total = 0;
    for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
        x = env2.step();
        float val = t->Step(x, env2.get_reward());
//            std::cout << "Val = " << val << " reward " << env2.get_reward() << std::endl;
        list_of_predictions.push_back(val);
        list_of_rewards.push_back(env2.get_reward());
        steps.push_back(i);
        if (i % 3000 == 0 && i > 10) {

            int s = list_of_predictions.size();
            std::vector<float> list_of_returns(s, 0);
            list_of_returns[s - 1] = 0;
            // compute returns; ignore returns for last 200 values
            float return_val = 0;
            for (int j = list_of_rewards.size() - 1; j >= 1; j--) {
                return_val = list_of_rewards[j] + env2.get_gamma() * return_val;
                list_of_returns[j - 1] = return_val;
            }

            for (int j = 0; j < s - 300; j++) {
                total += 1;
                error += (list_of_returns[j] - list_of_predictions[j]) *
                         (list_of_returns[j] - list_of_predictions[j]);
            }
            lifetime_error_metric.record_value(
                    {std::to_string(my_experiment->get_int_param("run")), std::to_string(total),
                     std::to_string(error / total)});


            if (i > (my_experiment->get_int_param("steps") - 5500) && list_of_predictions.size() > 2000) {
                for (int j = 0; j < 1000; j++) {
//                        std::cout << "recording pred\n";
                    predictions.record_value(
                            {std::to_string(my_experiment->get_int_param("run")), std::to_string(steps[j]),
                             std::to_string(list_of_predictions[j]), std::to_string(list_of_rewards[j]),
                             std::to_string(list_of_returns[j])});
                }
            }
            // remove everything but the last 300 values
            list_of_predictions.erase(list_of_predictions.begin(), list_of_predictions.begin() + s - 300);
            list_of_rewards.erase(list_of_rewards.begin(), list_of_rewards.begin() + s - 300);
            steps.erase(steps.begin(), steps.begin() + s - 300);


//            std::cout << i << "\tError " << error << "\n";
        }
        if ((i % 10000) == 0) {
            std::cout << "Committing values\n";
            lifetime_error_metric.commit_values();
            predictions.commit_values();
        }

    }
    lifetime_error_metric.commit_values();
    predictions.commit_values();
    // get step_sizes
    if (my_experiment->get_string_param("learner") == "SwiftTD") {
        SwiftTD *pointer = static_cast<SwiftTD *>(t);
        std::vector<float> step_size_per_pixel = pointer->GetStepSizePerPixel();
        for (int i = 0; i < step_size_per_pixel.size(); i++) {
            step_sizes.record_value(
                    {std::to_string(my_experiment->get_int_param("run")), std::to_string(i),
                     std::to_string(step_size_per_pixel[i])});
        }
        step_sizes.commit_values();
    }
    return 0;
}