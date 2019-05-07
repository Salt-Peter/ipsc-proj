#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <omp.h>
#include <vector>
#include <algorithm>
#include "LogisticRegression.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd one_vs_all_sequential(MatrixXd df);

VectorXd one_vs_all_parallel(MatrixXd df);

int main() {
    std::string filepath = "data/train_orig.csv";

    auto df = read_csv(filepath);
    std::cout << "df dimensions: " << df.rows() << "x" << df.cols() << std::endl;
    auto df_X_y = split_X_y(df);
    auto X = df_X_y.first;
    auto y = df_X_y.second;

    std::cout << "\nSequential:" << std::endl;
    double startTime = omp_get_wtime();
    auto y_pred = one_vs_all_sequential(df);
    double stopTime = omp_get_wtime();
    std::cout << "TIME: " << stopTime - startTime << std::endl;
    std::cout << "accuracy: " << accuracy(y_pred, y) << std::endl;

    std::cout << "\nParallel:" << std::endl;
    startTime = omp_get_wtime();
    y_pred = one_vs_all_parallel(df);
    stopTime = omp_get_wtime();
    std::cout << "TIME: " << stopTime - startTime << std::endl;
    std::cout << "accuracy: " << accuracy(y_pred, y) << std::endl;

    return 0;
}


VectorXd one_vs_all_sequential(MatrixXd df) {
    int label_col_index = df.cols() - 1;
    auto unique_labels = get_unique_labels(df, label_col_index);

    auto n_unique_labels = unique_labels.size();
    auto n_features = df.cols(); // -1 for label column and +1 for x0 column (column of ones)
    auto m = df.rows();

    // dim(label_probs) = m x num_unique_labels + 1(for storing the final highest probability label)
    Eigen::MatrixXd label_probs;
    label_probs.setZero(m, n_unique_labels + 1);


    for (auto i = 0; i < n_unique_labels; i++) {
        // convert multiclass to binary class
        auto df_bin = convert_to_binary_class(df, unique_labels[i]);

        auto df_X_y = split_X_y(df_bin);
        auto X = df_X_y.first;
        auto y = df_X_y.second;

        LogisticRegression lr(1e-5, 1000);
        VectorXd res; // contains prob


        lr.fit_naive(std::make_pair(X, y));
        res = lr.predict(X);


//        cout << "label_probs before: \n" << label_probs << endl;

        // ith column refers to ith model (in total there will be "n_unique_label" number of models)
        label_probs.col(i) << res;

//        cout << "label_probs after: \n" << label_probs << endl;
    }

    for (auto r = 0; r < label_probs.rows(); r++) {
        // for each row, set label to the class label with highest probability.
        label_probs(r, label_probs.cols() - 1) = unique_labels[max_index(label_probs.row(r))];
    }

//    std::cout << "label_probs Final sequential: \n" << label_probs << std::endl;

    Eigen::VectorXd y_pred = label_probs.col(label_probs.cols() - 1);

    return y_pred;

}

VectorXd one_vs_all_parallel(MatrixXd df) {
    int label_col_index = df.cols() - 1;
    auto unique_labels = get_unique_labels(df, label_col_index);

    auto n_unique_labels = unique_labels.size();
    auto n_features = df.cols(); // -1 for label column and +1 for x0 column (column of ones)
    auto m = df.rows();

    // dim(label_probs) = m x num_unique_labels + 1(for storing the final highest probability label)
    Eigen::MatrixXd label_probs;
    label_probs.setZero(m, n_unique_labels + 1);

#pragma omp parallel for
    for (auto i = 0; i < n_unique_labels; i++) {
        // convert multiclass to binary class
        auto df_bin = convert_to_binary_class(df, unique_labels[i]);

        auto df_X_y = split_X_y(df_bin);
        auto X = df_X_y.first;
        auto y = df_X_y.second;

        LogisticRegression lr(1e-5, 1000);
        VectorXd res; // contains prob


        lr.fit_parallel(std::make_pair(X, y));
        res = lr.predict(X);
//        cout << "label_probs before: \n" << label_probs << endl;
        // ith column refers to ith model (in total there will be "n_unique_label" number of models)
        label_probs.col(i) << res;
//        cout << "label_probs after: \n" << label_probs << endl;
    }

    for (auto r = 0; r < label_probs.rows(); r++) {
        // for each row, set label to the class label with highest probability.
        label_probs(r, label_probs.cols() - 1) = unique_labels[max_index(label_probs.row(r))];
    }

//    std::cout << "label_probs Final sequential: \n" << label_probs << std::endl;
    Eigen::VectorXd y_pred = label_probs.col(label_probs.cols() - 1);
    return y_pred;
}