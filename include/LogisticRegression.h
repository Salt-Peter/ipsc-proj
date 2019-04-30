#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <Eigen/Core>
#include <omp.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <numeric>
#include <Eigen/Core>
#include <util.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class LogisticRegression {
private:
    double alpha;
    double epsilon;
    int maxIter;

    VectorXd theta;

    long m;
    long n;

    bool verbose = false;

public:
    LogisticRegression(double learning_rate, int num_iters) {
        this->alpha = learning_rate;
        this->maxIter = num_iters;
    }


    VectorXd predict(MatrixXd test) {
        return (test * theta).unaryExpr(std::ptr_fun(g));
    }

    void fit_naive(std::pair<MatrixXd, VectorXd> train) {
        m = train.first.rows();
        n = train.first.cols();

        theta.setZero(n);

        for (size_t iter = 0; iter < maxIter; iter++) {

            VectorXd gw;
            gw.setZero(n);
            for (size_t i = 0; i < m; i++) {
                double coeff = train.first.row(i) * theta;

                // h(xi) = 1/m * (g(xi * theta) - yi)
                coeff = 1.0 / m * (g(coeff) - train.second(i));

                // gwi = h(xi) * xi
                VectorXd gwi = coeff * train.first.row(i);

                gw += gwi;
            }


            // theta -= alpha * gw
            this->theta -= this->alpha * gw;

            // check for convergence
            if (this->epsilon > gw.norm()) {
                break;
            }
        }
    }

    Eigen::MatrixXd _prefix_ones_column(Eigen::MatrixXd X) {
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


    void fit_sequential(Eigen::MatrixXd X, Eigen::VectorXd y) {
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

    void _gradient_descent(Eigen::MatrixXd X, Eigen::VectorXd y) {
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

        for (auto it = 0; it < maxIter; it++) {
            Eigen::VectorXd grad;
            grad.setZero(n);

            for (auto i = 0; i < m; i++) {
                auto h = g(X.row(i) * theta);
                auto error = (1.0 / m) * (h - y(i));

                Eigen::VectorXd gradi = error * X.row(i);
                grad += gradi;
            }

            this->theta = this->theta - (this->alpha * grad);


            // TODO: Add convergence test
        }
    }


    void fit_parallel(std::pair<MatrixXd, VectorXd> train) {
        m = train.first.rows();
        n = train.first.cols();

        theta.setZero(n);

        int coreNum = omp_get_num_procs();

        // temp sum in each processor
        std::vector<VectorXd> sumInCore(coreNum);

        // iterator in each processor
        std::vector<size_t> iters(coreNum);

        // iter count for each processor
        size_t sectionNum = m / coreNum;

        for (size_t iter = 0; iter < maxIter; iter++) {

            for (auto &item : sumInCore) {
                item.setZero(n);
            }

            VectorXd gw;
            gw.setZero(n);

#pragma omp parallel for
            for (size_t core = 0; core < coreNum; core++) {

                for (iters[core] = core * sectionNum;
                     iters[core] < (core + 1) * sectionNum && iters[core] < m; iters[core]++) {

                    double coeff = train.first.row(iters[core]) * theta;

                    // h(xi) = 1/m * (g(xi * theta) - yi)
                    coeff = 1.0 / m * (g(coeff) - train.second(iters[core]));

                    // gwi = h(xi) * xi
                    sumInCore[core] += coeff * train.first.row(iters[core]);

                }

            }

            // merge
            gw = std::accumulate(sumInCore.begin(), sumInCore.end(), gw);


            // theta -= alpha * gw
            this->theta -= this->alpha * gw;

            // check for convergence
            if (this->epsilon > gw.norm()) {
                break;
            }
        }
    }
};


#endif //LOGISTICREGRESSION_H