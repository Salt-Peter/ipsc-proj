#include <iostream>
#include "util.h"
#include "LogisticRegression.h"

using namespace std;

int main(int argc, char **argv) {
    string filepath;
    if (argc != 2) {
        // filepath not supplied in command line args
        // STOPSHIP
        filepath = "/home/abhinav/projects/ipsc-proj/data/train.csv";
//        return -1;
    } else {
        filepath = argv[1];
    }

    auto df = read_csv(filepath);
    cout << "df dimensions: " << df.rows() << "x" << df.cols() << endl;

    auto df_X_y = split_X_y(df);
    auto X = df_X_y.first;
    auto y = df_X_y.second;

    // TODO: Split df into train and validate part


    LogisticRegression model = LogisticRegression(0.1, 200, 0);

    // TODO: Fit model through train df
    model.fit_sequential(X, y);

    // TODO: predict over validate df
    auto X_test = read_csv("/home/abhinav/projects/ipsc-proj/data/test.csv");
    auto y_pred = model.predict(X_test);

    cout << y_pred << endl;
    return 0;
}