#ifndef NORMAL_LOGISTIC_REGRESSION
#define NORMAL_LOGISTIC_REGRESSION


#include<bits/stdc++.h>

class normal_logistic_regression{

    private:
    
    double alpha // learning rate
    data *dt//data is class containing data

    public:
    normal_logistic_regression (const data &d,alpha){
        // initialize parameters
        this->dt=d
        this->alpha=alpha
    }

    void train(){
        // start training on this->data->train_data
    }

    void validate(){
        // predict labels on this->data->test_data and find accuracy of model
    }

    void predict(const input &d){
        // predict labels on this->data->test_data and find accuracy of model
    }

};

#endif