#include <iostream>
#include <cmath>
#include <vector>

using namespace std;


class Activation { 
    /*
    (i) function definitions for multiple activation functions and their derivatives
    (ii) setter functions, used in the constructor, that set the desires activation function as the actual activation function
    (iii) getter functions that reference the activation function and its derivative and compute desired values
        ^^ these will be referenced in the layer class
    */
};

class Weight {
    private:
    public:
        vector<double> weights; // dim = [next.nodes,prev.nodes] ... use prev.nodes as spacer for referencing values (e.g. idx = [row]*prev.nodex + [col])
        Layer prev_layer;
        Layer next_layer;
        vector<double> bias; // 1d vector of size next_layer.nodes

        // constructor
        Weight(Layer _prev, Layer _next) {
            prev_layer = _prev;
            next_layer = _next;

            for (int _ = 0; _ < prev_layer.nodes*next_layer.nodes; _++) {
                weights.push_back(.1); // initial weight value, can randomize or make distribution rather than single value
                if (_ < next_layer.nodes) {bias.push_back(.1);} // initial bias value 
            }
        }

        // forward propagation
        vector<double> computeInput(vector<double> output) { // output is generated in the Layer class through the activation function ... output = activation(input)
            vector<double> output;
            
            for (int row = 0; row < next_layer.nodes; row++) {
                double tmp = 0;

                for (int col = 0; col < prev_layer.nodes; col++) {
                    tmp += weights[row*prev_layer.nodes + col] * output[col] + bias[col];
                }

                output.push_back(tmp);
                tmp = 0;
            }
        }

        // back propagation will be defined, computed, and re-assigning all in model class (since it requires information from other layers/weights)
};

class Layer { // parent class for (i) hidden layers ... (a) fully connected (b) not-fully (c) etc ... (ii) input layers (iii) output layers
    private:
    public:
        int nodes;

};

class HiddenLayer : Layer { // can have children (e.g. fully-connected/dense, sparsely-connected)

};

class InputLayer : Layer {

};

class OutputLayer : Layer {

};

int main() {
    return 0;
}