#include <iostream> 
#include <vector>
#include <cmath>

#include "model_class.h"
#include "model_component_classes.h"

using namespace std;

int main() {
    cout.flush();
    srand(time(NULL));

    // Xor data (if x or y, but not both, is 1 then 1 else 0)
    vector<double> x_train = {0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0}; 
    vector<double> y_train = {0.0,1.0,1.0,0.0};
    int num_features = 2;
    
    // making Layers and stuff vars as test of fixing "undefined ref" error
    Layer input_layer = Layer(num_features,"sigmoid");
    Layer hidden_layer = Layer(4,"sigmoid");
    Layer output_layer = Layer(1,"sigmoid");

    // build model
    Model model;
    model.add(input_layer,3); // input
    model.add(hidden_layer,3); // hidden
    model.add(output_layer,3); // output

    // compile model
    model.useLoss("mse");

    // test
    model.fit(x_train,y_train,20000,.1,num_features);
    model.predict(x_train,{});

    // evaluate
    cout << "Predictions\tProbabilities\tActuals" << endl;
    cout << "" << endl;
    for (int i = 0; i < size(x_train)/num_features; i++) {
        if (model.predictions[i] >= .5) {
            cout << 1 << "\t\t" << model.predictions[i] << "\t\t" << y_train[i] << endl;
        } else {
            cout << 0 << "\t\t" << model.predictions[i] << "\t\t" << y_train[i] << endl;
        }
    }
    cout << "" << endl;
    cout << "Final Loss: " << model.loss << endl;
    

    return 0;
}