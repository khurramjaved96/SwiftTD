//
// Created by Khurram Javed on 2025-03-13.
//
#include "SwiftTD.h"


int main(){
    SwiftTD* learner = new SwiftTDSparse(100, 0.99, 1e-8, 0.99, 1e-8, 0.25, 0.99, 1e-3);
    delete learner;
    return 0;
}
