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
    cout<<df.rows()<<endl;

    // TODO: Split df into train and validate part


    LogisticRegression model = LogisticRegression(0.1, 200, 0);

    // TODO: Fit model through train df
    model.fit_sequential(df);

    // TODO: predict over validate df
    model.predict(df);

    return 0;
}