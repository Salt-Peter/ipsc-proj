#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <Eigen/Core>

class LogisticRegression {
private:
    double alpha;   // learning rate
    int num_iters;
    double lambda;  // regularization constant

    double threshold = 0.5;

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
        _gradient_descent(X, y);
    }

    void _gradient_descent(Eigen::MatrixXd &X, Eigen::VectorXd &y) {
        /**
            theta = n+1 x 1
            alpha = scalar
            X = m x (n+1)
            h = m x 1
            y = m x 1
            -----------------------------------------
            theta = theta - (alpha/m) * X.T * (h - y)
         */

        auto m = X.rows(); // number of samples
        auto n = X.cols(); // number of samples

        for (auto it = 0; it < num_iters; it++) {
            Eigen::VectorXd grad;
            grad.setZero(n);

            for (auto i = 0; i < m; i++) {
                auto h = sigmoid(X.row(i) * theta);
                auto error = (1.0 / m) * (h - y(i));

                Eigen::VectorXd gradi = error * X.row(i);
                grad += gradi;
            }

            this->theta = this->theta - (this->alpha * grad);


            // TODO: Add convergence test
        }
    }


    void fit_parallel(Eigen::MatrixXd) {
        cout << "Fit parallel !!!Not yet implemented!!!" << endl;
    }


    Eigen::VectorXd predict(Eigen::MatrixXd &X) {
        auto y_prob = predict_prob(X);

        Eigen::VectorXd y_pred(y_prob.rows());

        // Convert probability to class label
        for (auto i = 0; i < y_pred.rows(); i++) {
            if (y_prob(i) >= threshold)
                y_pred(i) = 1;
            else
                y_pred(i) = 0;
        }

        return y_pred;
    }

    Eigen::VectorXd predict_prob(Eigen::MatrixXd X) {
        // TODO: normalize

        // dim(X) = m x n
        // add x0 column containing all ones
        X = this->_prefix_ones_column(X);

        // dim(X) = m x n+1 ; dim(theta) = n+1 x 1
        // y_prob = sigmoid(np.dot(X.values, self.theta))
        cout << "dim X: " << X.rows() << " x " << X.cols() << endl;
        auto y_prob = (X * this->theta).unaryExpr(std::ptr_fun(sigmoid)); // FIXME: convert to loop
        return y_prob;
//        return Eigen::VectorXd(X.cols() + 1);
    }
};


#endif //LOGISTICREGRESSION_H
