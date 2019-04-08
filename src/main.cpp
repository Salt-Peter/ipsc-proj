#include <iostream>
#include "util.h"

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
    cout << "dataframe :-\n" << df << endl;
    return 0;
}