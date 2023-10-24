#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Model {
    private:                    // NN requires at least 2 layers (input and output) 
        vector<Layer> layers; // e.g.   input       hidden      hidden      output
        vector<Weight> weights; // e.g.         W0          W1          W2
        vector<vector<double>> deltas; // ive been avoiding this but each delta is a different size so useful here
    public:
        vector<double> predictions;
        vector<double> actuals;
        vector<double> input_data; 
        double learning_rate;
        // Layer object has an Activation object as an attribute, so no need for activations vector

        Model(double _lr) {
            // not quite sure what the constructor needs to look like yet
            learning_rate = _lr;
        }

        // propagation functions
        vector<double> computeDeltas(int layer_num) {
            vector<double> tmp;

            if (layer_num == size(layers)-1){
                for (int i = 0; i < size(predictions); i++) {
                    tmp.push_back(predictions[i]-actuals[i]);
                }
                deltas.insert(deltas.begin(),tmp);
                return tmp;
            } else {
                // recursive computation of deltas
                // terminating condition established as above
            }
        } 

        void forwardPropagation() {
            vector<double> tmp_output;
            for (int i = 0; i < size(layers) - 1; i++) {
                tmp_output = layers[i].computeOutput();
                layers[i+1].inputs = weights[i].computeInput(tmp_output);
            }
            predictions = layers[size(layers)-1].computeOutput();
        }

        void backPropagation() {
            for (int i = 0; i < size(weights); i++) {
                weights[i].backPropagationWeights(learning_rate, , layers[i]);
                weights[i].backPropagationBias(learning_rate, );
            }
        }
};

int main() {
    return 0;
}