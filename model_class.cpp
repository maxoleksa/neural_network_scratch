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
        vector<double> predictions; // want to allow for multiple output nodes so not declaring single-type
        vector<double> actuals;
        vector<double> input_data; 
        double learning_rate;
        // Layer object has an Activation object as an attribute, so no need for activations vector

        Model(double _lr) {
            // not quite sure what the constructor needs to look like yet
            learning_rate = _lr;
        }

        // propagation functions
        vector<double> computeDeltas(int layer_num) { // recursive 
            vector<double> delta;
            double tmp_val = 0;

            if (layer_num == size(layers)-1){
                for (int i = 0; i < size(predictions); i++) {
                    delta.insert(delta.begin(),predictions[i]-actuals[i]);
                }
            } else {
                int prev_nodes = layers[layer_num].nodes;
                int next_nodes = layers[layer_num+1].nodes;

                for (int col = 0; col < prev_nodes; col++) { // transpose of weight matrix is used in calculation
                    for (int row = 0; row < next_nodes; row++){ // so row/col are backwards in loops
                        tmp_val += weights[layer_num+1][col*next_nodes + row] * computeDeltas(layer_num+1)[row];
                    }
                    delta.push_back(tmp_val*layers[layer_num].activation.pActivationDerivative(layers[layer_num].inputs[col]));
                    tmp_val = 0;
                }
            }
            deltas.insert(deltas.begin(),delta);
            return delta;
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
                weights[i].backPropagationWeights(learning_rate, deltas[i], layers[i]);
                weights[i].backPropagationBias(learning_rate, deltas[i]);
            }
        }
};

int main() {
    return 0;
}