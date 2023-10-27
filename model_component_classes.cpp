#include <iostream>
#include <cmath>
#include <vector>

//#include "model_component_classes.h"

using namespace std;

class Activation { 
    private:
        double (Activation::*pActivation) (double);
        double (Activation::*pActivationDerivative) (double);
    // various activation functions and their derivatives
        double sigmoid(double x) {
            return 1/(1+exp(-x));
        }
        double sigmoid_prime(double x) {
            return sigmoid(x)*(1-sigmoid(x));
        }

        double linear(double x) {
            return x;
        }
        double linear_prime(double _ = 0.0) {
            return 1;
        }

        double relu(double x) {
            return max(0.0,x);
        }
        double relu_prime(double x) {
            if (x <= 0.0) {return 0;} else {return 1;}
        }

        double hyper_tan(double x) {
            return tanh(x);
        }
        double hyper_tan_prime(double x) {
            return 1 - pow(tanh(x),2);
        }
    public:
        Activation(Activation &a) {
            pActivation = a.pActivation;
            pActivationDerivative = a.pActivationDerivative;
        }

        Activation() {
            pActivation=&sigmoid;
            pActivationDerivative=&sigmoid_prime;
        }

        Activation(string _activation) {
            if (_activation == "sigmoid") {pActivation=&sigmoid; pActivationDerivative=&sigmoid_prime;}
            else if (_activation == "linear") {pActivation=&linear; pActivationDerivative=&linear_prime;}
            else if (_activation == "relu") {pActivation=&relu; pActivationDerivative=&relu_prime;} 
            else {pActivation=&hyper_tan; pActivationDerivative=&hyper_tan_prime;}
        }

        void operator=(const Activation &a) {
            pActivation = a.pActivation;
            pActivationDerivative = a.pActivationDerivative;
        }

        double activationFunction(double x) {
            return (*this.*pActivation)(x);
        }
        double activationFunctionDerivative(double x) {
            return (*this.*pActivationDerivative)(x);
        }

        vector<double> generateOutputs (vector<double> inputs) { // "activate" the input data
            vector<double> outputs;
            for (int i = 0; i < size(inputs); i++) {
                outputs.push_back(activationFunction(inputs[i]));
            }
            return outputs;
        }

};

class Layer { // parent class for (i) hidden layers ... (a) fully connected (b) not-fully (c) etc ... (ii) input layers (iii) output layers
    private:                                            // (b) has not been implemented
        vector<double> inputs;
        vector<double> outputs;
    public:
        int nodes;
        Activation activation;

        Layer(Layer &l) {
            nodes = l.nodes;
            l.activation = l.activation;
        }

        Layer() {
            nodes = 3;
            activation = Activation();
        }

        Layer(int _nodes, string _activation) {
            nodes = _nodes;
            activation = Activation(_activation);
        }

        void operator=(const Layer &l) {
            inputs = l.inputs;
            outputs = l.outputs;
            nodes = l.nodes;
            activation = l.activation;
        }

        // getter functions for inputs and outputs
        void setInputs(vector<double> _inputs) {inputs = _inputs;}
        vector<double> getInputs(){return inputs;}
        vector<double> getOutputs() {return outputs;}

        vector<double> computeOutput() {
            outputs = activation.generateOutputs(inputs);
            return getOutputs();
        }
};

class Weight {
    private:
    public:
        vector<double> weights; // dim = [next.nodes,prev.nodes] ... use prev.nodes as spacer for referencing values (e.g. idx = [row]*prev.nodex + [col])
        Layer prev_layer;
        Layer next_layer;
        vector<double> bias; // 1d vector of size next_layer.nodes

        // constructors
        Weight(Weight &w) {
            weights = w.weights;
            prev_layer = w.prev_layer;
            next_layer = w.next_layer;
            bias = w.bias;
        }

        Weight() {
            // since default layer constructor uses 3 elements we will work under that assumption for the 
            // blank weight constructor
            for (int i = 0; i < 3; i++) {
                weights.push_back(.1);
                bias.push_back(.1);
            }
        }

        Weight(Layer _prev, Layer _next) {
            prev_layer = _prev;
            next_layer = _next;

            for (int _ = 0; _ < prev_layer.nodes*next_layer.nodes; _++) {
                weights.push_back(.1); // initial weight value, can randomize or make distribution rather than single value
                if (_ < next_layer.nodes) {bias.push_back(.1);} // initial bias value 
            }
        }

        void operator=(const Weight &w) {
            prev_layer = w.prev_layer;
            next_layer = w.next_layer;
            weights = w.weights;
            bias = w.bias;
        }

        // forward propagation
        vector<double> computeInput(vector<double> output) { // output is generated in the Layer class through the activation function ... output = activation(input)
            vector<double> input;
            
            for (int row = 0; row < next_layer.nodes; row++) {
                double tmp = 0;

                for (int col = 0; col < prev_layer.nodes; col++) {
                    tmp += weights[row*prev_layer.nodes + col] * output[col] + bias[col];
                }

                input.push_back(tmp);
                tmp = 0;
            }

            return input;
        }

        // back propagation 
        // delta relies on higher layer deltas and prev_outputs relies on the previous layer ... thus values must be stored in model class and then passed to this function for model back propagation
        void backPropagationWeights(double lr, vector<double> delta, vector<double> prev_outputs) {
            for (int i = 0; i < size(delta); i++) {
                for (int j = 0; j < size(prev_outputs); j++) {
                    weights[i*size(prev_outputs) + j] -= lr*delta[i]*prev_outputs[j];
                }
            }
        }
        void backPropagationBias(double lr, vector<double> delta) {
            for (int i = 0; i < size(delta); i++) {
                bias[i] -= lr*delta[i];
            }
        }
};


// a lot of functions will be stored in model class rather than components
// e.g. batch read of data requires averaging of model outputs so function will be defined there rather than in input sub-class (since need all individual outputs and then averaging them)

int main() {return 0;}