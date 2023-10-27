#include <iostream> 
#include <vector>

#include "model_class.h"
#include "model_component_classes.h"

using namespace std;

int main() {
    // Xor data (if x or y, but not both, is 1 then 1 else 0)
    vector<double> x_train = {0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0}; 
    vector<double> y_train = {0.0,1.0,1.0,0.0};
    int num_features = 2;

    // build model
    Model model;
    model.add(Layer(num_features,"hyper_tan")); // input
    model.add(Layer(3,"hyper_tan")); // hidden
    model.add(Layer(1,"hyper_tan")); // output

    // compile model
    model.useLoss("log");

    // test
    model.fit(x_train,y_train,100,.1,num_features);
    model.predict(x_train);

    // evaluate
    cout << "Predictions\tActuals" << endl;
    for (int i = 0; i < size(x_train)/num_features; i++) {
        cout << model.predictions[i] << "\t" << y_train[i] << endl;
    }

    return 0;
}