#ifndef UTIL_H
#define UTIL_H

#include <fstream> // ifstream, ofstream, fstream classes
#include "Eigen/Dense"
#include <algorithm>

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

#endif //UTIL_H