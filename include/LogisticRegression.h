#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <Eigen/Core>

class LogisticRegression {
private:
    double alpha;   // learning rate
    int num_iters;
    double lambda;  // regularization constant

    Eigen::VectorXd theta; // model coefficients


    Eigen::MatrixXd _prefix_ones_column(Eigen::MatrixXd &X) {
        auto row = X.rows();
        auto col = X.cols() + 1;
        // TODO: Find if we can prefix col instead of creating a new matrix.
        Eigen::MatrixXd X_modified(row, col);
        for (auto i = 0; i < row; i++) {
            // Add column(feature) x0 containing all ones
            X_modified(i, 0) = 1;
            for (auto j = 1; j < col; j++) {
                X_modified(i, j) = X(i, j - 1);
            }
        }

        return X_modified;
    }

public:
    LogisticRegression(double learning_rate, int num_iters, double regularization_constant = 0) {
        this->alpha = learning_rate;
        this->num_iters = num_iters;
        this->lambda = regularization_constant;
    }

    void fit_sequential(Eigen::MatrixXd &X, Eigen::VectorXd &y) {
        long m, n;
        m = X.rows();  // number of examples
        n = X.cols();  // number of features


        // TODO: normalize feature set X
        // X = _normalize(X)

        // dim(X) = m x n
        X = _prefix_ones_column(X);
        // dim(X) = m x n+1


        // initialize theta with some values
        theta.setRandom(n + 1); // dim(theta) = n+1 x 1

        // TODO: invoke SGD
        // dim(X) = m x n+1;  dim(y) = m x 1
        // _gradient_descent(X, y);

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
