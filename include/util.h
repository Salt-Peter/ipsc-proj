#ifndef UTIL_H
#define UTIL_H

#include <fstream> // ifstream, ofstream, fstream classes
#include "Eigen/Dense"
#include <algorithm>
#include <vector>

using namespace std;


pair<long, long> get_dimensions(const string &filepath) {
    ifstream f(filepath);
    assert(f.good());

    long rows = 0, cols = 0;

    string line;
    while (getline(f, line)) {

        if (line.empty() or (line.length() == 1 and isspace(line[0]))) // skip empty lines
            continue;

        rows += 1;

        if (rows == 1) {
            // First row read
            // Now calculate the number of columns
            long n = count(line.begin(), line.end(), ',');
            cols = n + 1;
        }
    }

    f.close();

    return make_pair(rows, cols);
}

Eigen::MatrixXd read_csv(const string &filepath) {
    ifstream f(filepath);
    assert(f.good());

    long rows = 0, cols = 0;

    auto dim = get_dimensions(filepath);
    rows = dim.first;
    cols = dim.second;


    Eigen::MatrixXd df(rows, cols);
    string line, word;

    for (auto i = 0; i < rows; i++) {
        getline(f, line);
        stringstream ss(line);

        for (auto j = 0; j < cols; j++) {
            getline(ss, word, ',');
            df(i, j) = stod(word);  // FIXME: handle reading STRING data in csv
        }
    }

    f.close();

    return df;
}


pair<Eigen::MatrixXd, Eigen::VectorXd> split_X_y(Eigen::MatrixXd df) {
    auto row = df.rows();
    auto col = df.cols();

    Eigen::MatrixXd X(row, col - 1);
    Eigen::VectorXd y(row);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col - 1; j++) {
            X(i, j) = df(i, j);
        }

        y(i) = df(i, col - 1);
    }

    return make_pair(X, y);
}


double sigmoid(double z) {
/** Sigmoid function :
 * g(z) = 1 / (1 + e^(-z))
 */

    return 1.0 / (1.0 + exp(-z));
}

Eigen::MatrixXd convert_to_binary_class(Eigen::MatrixXd &df, long true_class) {
    auto m = df.rows();
    auto label_index = df.cols() - 1;
    Eigen::MatrixXd df_bin = df;
    for (auto i = 0; i < m; i++) {
        if (df(i, label_index) == true_class)
            df_bin(i, label_index) = 1;
        else
            df_bin(i, label_index) = 0;
    }

    return df_bin;
}


vector<int> get_unique_labels(Eigen::MatrixXd &df, int label_index) {
    auto m = df.rows();
    vector<int> labels(m, 0);
    for (auto i = 0; i < m; i++) {
        labels[i] = df(i, label_index);
    }

    // {1,2,3,1,2,3,3,4,5,4,5,6,7};
    sort(labels.begin(), labels.end());
    // 1 1 2 2 3 3 3 4 4 5 5 6 7
    auto last = std::unique(labels.begin(), labels.end());
    // it now holds {1 2 3 4 5 6 7 x x x x x x}, where 'x' is indeterminate
    labels.erase(last, labels.end());

    return labels;
}

//// TODO: Find an efficient way or a library method
int max_index(Eigen::RowVectorXd row) {
    // Return index of max element
    int max_index = 0;
    for (auto i = 1; i < row.cols(); i++) {
        if (row(i) > row(max_index))
            max_index = i;
    }
    return max_index;
}

double accuracy(Eigen::VectorXd &y_pred, Eigen::VectorXd &y_true) {
    int correct_count = 0;
    for (auto i = 0; i < y_pred.rows(); i++) {
        if (y_pred.row(i) == y_true.row(i)) {
            correct_count++;
        }
    }
    return (correct_count * 1.0) / y_pred.rows();
}

#endif //UTIL_H
