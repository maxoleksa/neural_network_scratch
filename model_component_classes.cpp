#include <iostream>
#include <cmath>
#include <vector>

#include "model_component_classes.h"

using namespace std;

// Activation Class

// various activation functions and their derivatives
double Activation::sigmoid(double x) {
    return 1/(1+exp(-x));
}
double Activation::sigmoid_prime(double x) {
    return sigmoid(x)*(1-sigmoid(x));
}

double Activation::linear(double x) {
    return x;
}
double Activation::linear_prime(double _ = 0.0) {
    return 1;
}

double Activation::relu(double x) {
    return max(0.0,x);
}
double Activation::relu_prime(double x) {
    if (x <= 0.0) {return 0;} else {return 1;}
}

double Activation::hyper_tan(double x) {
    return tanh(x);
}
double Activation::hyper_tan_prime(double x) {
    return 1 - pow(tanh(x),2);
}

/* commenting %_ constructors as test for undefined reference error
Activation(Activation &a) {
    pActivation = a.pActivation;
    pActivationDerivative = a.pActivationDerivative;
}
*/
Activation::Activation() {
    pActivation=&sigmoid;
    pActivationDerivative=&sigmoid_prime;
}

Activation::Activation(string _activation) {
    if (_activation == "sigmoid") {pActivation=&sigmoid; pActivationDerivative=&sigmoid_prime;}
    else if (_activation == "linear") {pActivation=&linear; pActivationDerivative=&linear_prime;}
    else if (_activation == "relu") {pActivation=&relu; pActivationDerivative=&relu_prime;} 
    else {pActivation=&hyper_tan; pActivationDerivative=&hyper_tan_prime;}
}

void Activation::operator=(const Activation &a) {
    pActivation = a.pActivation;
    pActivationDerivative = a.pActivationDerivative;
}

double Activation::activationFunction(double x) {
    return (*this.*pActivation)(x);
}
double Activation::activationFunctionDerivative(double x) {
    return (*this.*pActivationDerivative)(x);
}

vector<double> Activation::generateOutputs (vector<double> inputs) { // "activate" the input data
    vector<double> outputs;
    for (int i = 0; i < size(inputs); i++) {
        outputs.push_back(activationFunction(inputs[i]));
    }
    return outputs;
}


// Layer class


/* commenting %_ constructors as test for undefined reference error
Layer(Layer &l) {
    nodes = l.nodes;
    l.activation = l.activation;
}
*/
Layer::Layer() {
    nodes = 3;
    activation = Activation();
}

Layer::Layer(int _nodes, string _activation) {
    nodes = _nodes;
    activation = Activation(_activation);
}

void Layer::operator=(const Layer &l) {
    inputs = l.inputs;
    outputs = l.outputs;
    nodes = l.nodes;
    activation = l.activation;
}

// getter and setter functions for inputs and outputs
void Layer::setInputs(vector<double> _inputs) {inputs = _inputs;}
vector<double> Layer::getInputs(){return inputs;}
vector<double> Layer::getOutputs() {return outputs;}

vector<double> Layer::computeOutput() {
    outputs = activation.generateOutputs(inputs);
    return getOutputs();
}


// Weight Class


// constructors

/*
Weight::Weight(Weight &w) {
    weights = w.weights;
    prev_layer = w.prev_layer;
    next_layer = w.next_layer;
    bias = w.bias;
}
*/
Weight::Weight() {
    // since default layer constructor uses 3 elements we will work under that assumption for the 
    // blank weight constructor
    for (int i = 0; i < 3; i++) {
        weights.push_back(.1);
        bias.push_back(.1);
    }
}

Weight::Weight(Layer _prev, Layer _next) {
    prev_layer = _prev;
    next_layer = _next;

    for (int _ = 0; _ < prev_layer.nodes*next_layer.nodes; _++) {
        weights.push_back(.1); // initial weight value, can randomize or make distribution rather than single value
        if (_ < next_layer.nodes) {bias.push_back(.1);} // initial bias value 
    }
}

void Weight::operator=(const Weight &w) {
    prev_layer = w.prev_layer;
    next_layer = w.next_layer;
    weights = w.weights;
    bias = w.bias;
}

// forward propagation
vector<double> Weight::computeInput(vector<double> output) { // output is generated in the Layer class through the activation function ... output = activation(input)
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
void Weight::backPropagationWeights(double lr, vector<double> delta, vector<double> prev_outputs) {
    for (int i = 0; i < size(delta); i++) {
        for (int j = 0; j < size(prev_outputs); j++) {
            weights[i*size(prev_outputs) + j] -= lr*delta[i]*prev_outputs[j];
        }
    }
}
void Weight::backPropagationBias(double lr, vector<double> delta) {
    for (int i = 0; i < size(delta); i++) {
        bias[i] -= lr*delta[i];
    }
}



// a lot of functions will be stored in model class rather than components
// e.g. batch read of data requires averaging of model outputs so function will be defined there rather than in input sub-class (since need all individual outputs and then averaging them)
