#include <iostream>
#include <vector>
#include <iomanip>
#include <limits>
#include <numbers>
#include <random>


int main() {
    // impleneting true online td(lambda)
    // data stream
    // 1 1 1 0 0 0
    // reward one after terminal srtate
    // return is always 1

    int total_steps = 10000;
    std::vector<int> list_of_online{3};
    for (auto &online: list_of_online) {
        const int feature_val = 1;
        std::vector<float> states;
        for (int i = 0; i < total_steps; i++) {
            if (i % 100 > 92 && i % 100 < 97)
                states.push_back(1);
            else
                states.push_back(0);
        }
        const int constant = 1;
        std::mt19937 gen{0};


        std::vector<float> rewards;
        for (int i = 0; i < total_steps; i++) {
            if (i % 100 == 99)
                rewards.push_back(1);
            else
                rewards.push_back(0);
        }

        float gamma = 0.9;
        float lambda = 1.0;

        std::vector<float> returns = std::vector<float>(rewards.size(), 0);

        float return_val = 0;
        for (int i = rewards.size() - 1; i >= 0; i--) {
            return_val = rewards[i] + gamma * return_val;
            returns[i] = return_val;
        }


        float weight = 0;
        float step_size = 0.00001;
        float meta_step_size = 0.05;

        std::normal_distribution<float> normal_distribution(0, 1);

        float beta = log(step_size);
        float h = 0;
        std::cout << "Epi\tW\t\t\tH\t\t\tBeta\tStep-size" << std::endl;
        if (online == -1) {
            for (int episode = 0; episode < 1; episode++) {
                float e = 0;
                float feature = states[0];
                for (int i = 1; i < states.size(); i++) {
                    float value = weight * feature;
                    float value_next = weight * states[i];
                    float td_error = rewards[i - 1] + gamma * value_next - value;
                    e = feature + e * gamma * lambda;
                    weight += step_size * (td_error) * e;
                    feature = states[i];
                    if (i < 200)
                        std::cout << std::setprecision(6) << episode << "\t" << weight << "\t" << h << "\t"
                                  << e
                                  << "\t" << step_size << std::endl;
//                    std::cout << "td_error " << td_error << std::endl;
//                    std::cout << "weight after ep " << weight << std::endl;
                }

            }
        } else if (online == 0) {
            for (int episode = 0; episode < 1; episode++) {
                float e = 0;
                float v_old = 0;
                float feature = states[0];
                h = 0;
                for (int i = 1; i < states.size(); i++) {
                    float value = weight * feature;
                    float value_next = weight * states[i];
                    float td_error = rewards[i - 1] + gamma * value_next - value;
                    e = feature + e * gamma * lambda - step_size * lambda * gamma * (e * feature) * feature;
                    weight += step_size * (td_error + value - v_old) * e - step_size * (value - v_old) * feature;
                    h = h * (1 - step_size * e * feature) + step_size * (td_error + value - v_old) * e -
                        step_size * (value - v_old) * feature;
                    v_old = value_next;
                    feature = states[i];
                    if (i < 200)
                        std::cout << std::setprecision(6) << i << "\t" << weight << "\t" << h << "\t"
                                  << e
                                  << "\t" << step_size << std::endl;

                }
            }
        } else if (online == 1) {
            for (int episode = 0; episode < 1; episode++) {
                float e = 0;
                float v_old = 0;
                float feature = states[0];
                h = 0;

                for (int i = 1; i < states.size(); i++) {
                    float value = weight * feature;
                    float value_next = weight * states[i];
                    float td_error = rewards[i - 1] + gamma * value_next - v_old;
                    beta += meta_step_size / step_size * td_error * h * e / step_size;
                    step_size = exp(beta);
                    e *= gamma * lambda;
                    e += feature * step_size - step_size * (e * feature) * feature;
                    weight += (td_error) * e - step_size * (value - v_old) * feature;
                    h = h * (1 - e * feature) + (td_error) * e - step_size * (value - v_old) * feature;;
                    v_old = value_next;
                    feature = states[i];
                    if (i < 200)
                        std::cout << std::setprecision(6) << i << "\t" << weight << "\t" << h << "\t"
                                  << beta
                                  << "\t" << step_size << std::endl;
//                std::cout << "weight after ep " << weight << std::endl;
                }

            }
        } else if (online == 2) {
            float be = 0;
            float old_h = 0;
            h = 0;
            for (int episode = 0; episode < 6000; episode++) {
                float e = 0;
                float v_old = 0;
                float feature = states[0];
                float accumulatd_updated = 0;
//                h = 0;
                for (int i = 1; i < states.size(); i++) {
                    float value = weight * feature;
                    float value_next = weight * states[i];
                    float td_error = rewards[i - 1] + gamma * value_next - v_old;
                    accumulatd_updated += td_error * h * feature;
                    beta += meta_step_size * td_error * h * e / step_size;
//                    std::cout << meta_step_size << " " << td_error << " " << h << " " << e << std::endl;
                    step_size = exp(beta);
                    be = lambda * gamma * be + step_size * feature -
                         gamma * lambda * step_size * e * feature * feature -
                         step_size * gamma * lambda * feature * feature * be;
                    e *= gamma * lambda;
                    e += feature * step_size - step_size * (e * feature) * feature;
                    weight += (td_error) * e - step_size * (value - v_old) * feature;

                    float temp = h;
                    h = h + td_error * be + e * (gamma * states[i] * h - old_h * feature) -
                        feature * (step_size * (value - v_old) + step_size * feature * (h - old_h));
                    old_h = temp;
                    v_old = value_next;
                    feature = states[i];
                    if (i < 2)
                        std::cout << std::setprecision(6) << i << "\t" << weight << "\t" << h << "\t"
                                  << beta
                                  << "\t" << accumulatd_updated << std::endl;
//                std::cout << "weight after ep " << weight << std::endl;
                }


            }
        } else if (online == 3) {

            float old_h = 0;
            h = 0;
            float gradient_norm = 0;
            float momentum_term = 0;
            float counter_t = 0;
            float e = 0;
            float be = 0;
            float v_old = 0;
            float feature = states[0];
            float accumulatd_updated = 0;
            float idbd_trace = 0;
            float error_sum = 0;
            float decay = 0.99;
            for (int i = 1; i < states.size(); i++) {
                counter_t++;
                float value = weight * feature;
                float value_next = weight * states[i];
                float td_error = rewards[i - 1] + gamma * value_next - v_old;
                float td_error_new = rewards[i - 1] + gamma * value_next - value;
                idbd_trace = idbd_trace * gamma * lambda + h * feature;
                accumulatd_updated += td_error * idbd_trace;
                gradient_norm = gradient_norm * decay + (1 - decay) * (td_error * idbd_trace) * (td_error * idbd_trace);
                momentum_term = decay * momentum_term + (1 - decay) * (td_error * idbd_trace);

                float corrected_momentum_term = momentum_term / (1 - pow(decay, counter_t));
                float corrected_gradient_norm = gradient_norm / (1 - pow(decay, counter_t));


                beta += meta_step_size * (corrected_momentum_term / (sqrt(corrected_gradient_norm) + 1e-8));

                step_size = exp(beta);
                be = lambda * gamma * be + step_size * feature -
                     gamma * lambda * step_size * e * feature * feature -
                     step_size * gamma * lambda * feature * feature * be;
                e *= gamma * lambda;
                e += feature * step_size - step_size * (e * feature) * feature;
                weight += (td_error) * e - step_size * (value - v_old) * feature;
                float temp = h;
                h = h + td_error * be + e * (gamma * states[i] * h - old_h * feature) -
                    feature * (step_size * (value - v_old) + step_size * feature * (h - old_h));
                old_h = temp;
                v_old = value_next;
                feature = states[i];
                std::cout << i << "\t" << value << " " << returns[i] << "\t" << weight << "\t" << h << "\t"
                          << beta
                          << "\t" << step_size << " " << error_sum << std::endl;
//                std::cout << "Returns = " << returns[i] << std::endl;
                error_sum += (returns[i - 1] - value) * (returns[i - 1] - value);

            }

        } else if (online == 5) {
            for (int episode = 0; episode < 20; episode++) {
                float feature = states[0];
                float accumulated_update = 0;
//                h = 0;
                for (int i = 1; i < states.size(); i++) {
                    float value = weight * feature;
                    float td_error = rewards[i - 1] + gamma * returns[i] - value;
//                    counter_square++;
//                    square_runny_estimate += td_error * h * feature * td_error * h * feature;
                    accumulated_update += td_error * h * feature;
//                    beta += (meta_step_size * counter_square / (sqrt(square_runny_estimate) + 1e-8)) * td_error * h *
//                            feature;
//                    step_size = exp(beta);
                    beta += meta_step_size / step_size * td_error * h * feature;
                    step_size = exp(beta);
                    weight += step_size * ((td_error) * feature);
                    h = h * (1 - step_size * feature * feature) + step_size * td_error * feature;
                    feature = states[i];
                    if (i < 2)
                        std::cout << std::setprecision(6) << episode << " " << i << "\t" << weight << "\t" << h << "\t"
                                  << beta
                                  << "\t" << step_size << std::endl;
                }
//            weight = temp_w;


            }
        }
    }

    // offline lambda returns
    // 1 1 1 0 0 0
    // reward one after terminal srtate
    // return is always 1


    std::cout << "Hello, World!" << std::endl;
    return 0;
}


