#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <Eigen/Core>

class LogisticRegression {
private:
    double alpha;   // learning rate
    int num_iters;
    double lambda;  // regularization constant

public:
    LogisticRegression(double learning_rate, int num_iters, double regularization_constant = 0) {
        this->alpha = learning_rate;
        this->num_iters = num_iters;
        this->lambda = regularization_constant;
    }

    void fit_sequential(Eigen::MatrixXd) {
        cout << "Fit sequential !!!Not yet implemented!!!" << endl;
    }


    void fit_parallel(Eigen::MatrixXd) {
        cout << "Fit parallel !!!Not yet implemented!!!" << endl;
    }


    Eigen::VectorXd predict(Eigen::MatrixXd) {
        cout << "Predict !!!Not yet implemented!!!" << endl;
        return Eigen::VectorXd();
    }

};


#endif //LOGISTICREGRESSION_H
