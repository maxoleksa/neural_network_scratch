#include <iostream> 
#include <vector>

#include "model_class.h"
#include "model_component_classes.h"

using namespace std;

int main() {
    cout.flush();
    // Xor data (if x or y, but not both, is 1 then 1 else 0)
    vector<double> x_train = {0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0}; 
    vector<double> y_train = {0.0,1.0,1.0,0.0};
    int num_features = 2;
    
    // making Layers and stuff vars as test of fixing "undefined ref" error
    Layer input_layer = Layer(num_features,"hyper_tan");
    Layer hidden_layer = Layer(3,"hyper_tan");
    Layer output_layer = Layer(1,"hyper_tan");

    // build model
    Model model;
    model.add(input_layer); // input
    model.add(hidden_layer); // hidden
    model.add(output_layer); // output

    // compile model
    model.useLoss("log");

    // test
    model.fit(x_train,y_train,100,.1,num_features);
    model.predict(x_train);

    // evaluate
    cout << "Predictions\tProbabilities\tActuals" << endl;
    cout << "" << endl;
    for (int i = 0; i < size(x_train)/num_features; i++) {
        if (model.predictions[i] >= .5) {
            cout << 1 << "\t" << model.predictions[i] << "\t" << y_train[i] << endl;
        } else {
            cout << 0 << "\t" << model.predictions[i] << "\t" << y_train[i] << endl;
        }
    }
    
    return 0;
}