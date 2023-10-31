#include <iostream>
#include <cmath>
#include <vector>

#include "model_component_classes.h"
#include "model_class.h"

using namespace std;


// various loss functions (has a lot of room for expansion)
// these loss functions, and the Activation activation functions, can be moved to separate files
// but since only a few are included they are defined in class
double Model::binaryCrossEntropy(vector<double> preds, vector<double> acts) {
    double sm = 0.0;
    for (int i = 0; i < size(preds); i++) {
        sm += acts[i]*log(max(preds[i],pow(10,-9))) + (1-acts[i])*(log(1-max(preds[i],pow(10,-9))));
    }
    return -1.0/size(preds)*sm;
}
double Model::logLoss(vector<double> preds, vector<double> acts) { // classification
    double sm = 0.0;
    for (int i = 0; i < size(preds); i++) {
        sm += acts[i]*log(max(preds[i],pow(10,-9)));
    }
    return -1.0/size(preds)*sm;
}
double Model::mse(vector<double> preds, vector<double> acts) { // regression
    double sm = 0.0;
    for (int i = 0; i < size(preds); i++) {
        sm += pow(preds[i] - acts[i],2);
    }
    return 1.0/size(preds)*sm;
}
double Model::mae(vector<double> preds, vector<double> acts) {
    double sm = 0.0;
    for (int i = 0; i < size(preds); i++) {
        sm += abs(preds[i] - acts[i]);
    }
    return 1.0/size(preds)*sm;
}

Model::Model() {
    // constructor is blank since anything that we would define here we prefer to define through functions
}

// functions for building the model

// adding layers
void Model::add(Layer _layer, double weight_range) {
    layers.push_back(_layer);
    if (size(layers) > 1) {
        weights.push_back(Weight(layers[size(layers)-2],layers[size(layers)-1],weight_range));
    }
}
// setter for loss function
void Model::useLoss(string _loss) {
    // classification
    if (_loss == "log") {pLoss = &logLoss;} 
    else if (_loss == "binary cross-entropy") {pLoss = &binaryCrossEntropy;}
    // regression
    else if (_loss == "mse") {pLoss = &mse;}
    else {pLoss = &mae;}
}

// propagation functions

// deltas for backPropagation
void Model::computeDeltas(double actual) { 
    deltas = {};
    for (int layer_num = size(layers)-1; layer_num > 0; layer_num--){ // delta0 is never used so leave out
        vector<double> delta;
        double tmp_val = 0;

        if (layer_num == size(layers)-1){
            for (int i = 0; i < size(predictions); i++) {
                delta.insert(delta.begin(), predictions[i]-actual);
            }
        } else {
            int prev_nodes = layers[layer_num].nodes;
            int next_nodes = layers[layer_num+1].nodes;

            for (int col = 0; col < prev_nodes; col++) { // transpose of weight matrix is used in calculation
                for (int row = 0; row < next_nodes; row++){ // so row/col are backwards in loops
                    tmp_val += weights[layer_num].weights[row * prev_nodes + col] * deltas[0][row]; // effectively creating a memo recursive function since deltas is stored in class 
                }                                                                           // could have kept it recursive by taking 'computeDeltas[row]' outside of for loops and storing in var                                                                                      
                delta.push_back(tmp_val*layers[layer_num].activation.activationFunctionDerivative(layers[layer_num].getInputs()[col]));
                tmp_val = 0;
            }
        }
        deltas.insert(deltas.begin(),delta);
    }
} 

double Model::calculateLoss(vector<double> preds, vector<double> acts) {
    loss = (*this.*pLoss)(preds,acts);
    return loss;
}

double Model::predict(vector<double> input, double actual) {
    if (size(input) == num_features) { // forward prop for single data point (fitting)
        layers[0].setInputs(input);
        for (int i = 0; i < size(layers) - 1; i++) { 
            layers[i+1].setInputs( weights[i].computeInput( layers[i].computeOutput() ) );
        }
        predictions = layers[size(layers)-1].computeOutput();
        return calculateLoss(predictions,{actual});

    } else { // forward prop for multiple datapoints (final prediction)
        predictions = {};
        for (int j = 0; j < size(input)/num_features; j++) { 
            vector<double> tmp_input;
            vector<double> tmp_pred;

            for (int k = 0; k < num_features; k++) { // get all features in datapoint
                tmp_input.push_back(input[j*num_features + k]);
            }

            layers[0].setInputs(tmp_input); 
            for (int i = 0; i < size(layers) - 1; i++) { 
                layers[i+1].setInputs( weights[i].computeInput( layers[i].computeOutput() ) );
            }    

            tmp_pred = layers[size(layers)-1].computeOutput(); // add prediction to final predictions attribute
            for (int _ = 0; _ < size(tmp_pred); _++) {
                predictions.push_back(tmp_pred[_]);
            }
        }
        
        return calculateLoss(predictions,actuals);  
    }
}

void Model::backPropagation(double learning_rate, double actual) {
    computeDeltas(actual);
    for (int i = 0; i < size(weights); i++) {
        weights[i].backPropagationWeights(learning_rate, deltas[i], layers[i].getOutputs());
        weights[i].backPropagationBias(learning_rate, deltas[i]);
    }
}

void Model::fit(vector<double> x_train, vector<double> y_train, int epochs, double lr, int num_feats) {
    input_data = x_train;
    num_features = num_feats;
    actuals = y_train;
    for (int i = 0; i < epochs; i++) {
        double error = 0.0;
        double avg_epoch_error;
        for (int j = 0; j < size(x_train)/num_feats; j++) {
            vector<double> tmp_vec;
            for (int k = 0; k < num_feats; k++) {
                tmp_vec.push_back(x_train[j*num_feats + k]);
            }

            error += predict(tmp_vec,actuals[j]); // needs to be generalized
            backPropagation(lr,actuals[j]); // same with x_train, y_train (should be vector<vector<>> instead of 1d go allow for higher dimensional data)
        }

        avg_epoch_error = error / (size(x_train)/num_feats);
        cout << "Epoch " << i+1 << '/' << epochs << "\tLoss = " << avg_epoch_error << endl;
        cout << endl;
    }
}
