//
// Created by Khurram Javed on 2025-03-13.
//
#include "SwiftTD.h"

int main(){
    SwiftTDSparse* learner = new SwiftTDSparse(100, 0.99, 1e-8, 0.99, 1e-8, 0.25, 0.99, 1e-3);
    auto prediction = learner->Step({1, 2, 3, 4, 5}, 1.0);
    delete learner;
    return 0;
}
