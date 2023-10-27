#include <iostream> 
#include <vector>

using namespace std;

int main() {
    // Xor data (if x or y, but not both, is 1 then 1 else 0)
    vector<double> x_train = {0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0}; 
    vector<double> y_train = {0.0,1.0,1.0,0.0};
    int num_features = 2;

    // build model
    model = Model();
    model.add(Layer(num_features,"hyper_tan")); // input
    model.add(Layer(3,"hyper_tan")); // hidden
    model.add(Layer(1,"hyper_tan")); // output

    // compile model
    model.useLoss("log");
    model.fit(x_train,y_train,10,.1,num_features);

    // test
    model.predict(x_train);
    cout << model.predictions;

    return 0;
}