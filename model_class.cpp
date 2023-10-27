#include <iostream>
#include <cmath>
#include <vector>

//#include "model_class.h"
#include "model_component_classes.h"

using namespace std;

class Model {
    private:                    // NN requires at least 2 layers (input and output) 
        vector<Layer> layers; // e.g.   input       hidden      hidden      output
        vector<Weight> weights; // e.g.         W0          W1          W2
        vector<vector<double>> deltas; // ive been avoiding this but each delta is a different size so useful here
        int num_features;
        double (Model::*pLoss) (vector<double>, vector<double>);

    // various loss functions (has a lot of room for expansion)
    // these loss functions, and the Activation activation functions, can be moved to separate files
    // but since only a few are included they are defined in class
        double logLoss(vector<double> preds, vector<double> acts) { // classification
            double sm = 0.0;
            for (int i = 0; i < size(preds); i++) {
                sm += acts[i]*log(preds[i]);
            }
            return -1/size(preds)*sm;
        }
        double mse(vector<double> preds, vector<double> acts) { // regression
            double sm = 0.0;
            for (int i = 0; i < size(preds); i++) {
                sm += pow(preds[i] - acts[i],2);
            }
            return 1/size(preds)*sm;
        }
    public:
        vector<double> predictions; // want to allow for multiple output nodes so not declaring single-type
        vector<double> actuals;
        vector<double> input_data;
        double learning_rate;
        double loss;
        // Layer object has an Activation object as an attribute, so no need for activations vector

        Model() {
            // constructor is blank since anything that we would define here we prefer to define through functions
        }

        // functions for building the model

        // adding layers
        void add(Layer _layer) {
            layers.push_back(_layer);
            if (size(layers) > 1) {
                weights.push_back(Weight(layers[size(layers)-2],layers.back())); // .back returns [size() - 1] so 2nd to last is [size() - 2]
            }
        }
        // setter for loss function
        void useLoss(string _loss) {
            // classification
            if (_loss == "binary cross-entropy" || _loss == "log") {pLoss = &logLoss;} 
            // regression
            else {pLoss = &mse;}
        }

        // propagation functions

        // deltas for backPropagation
        void computeDeltas() { 
            deltas = {{}};
            for (int layer_num = size(layers)-1; layer_num >= 0; layer_num--){
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
                            tmp_val += weights[layer_num+1].weights[col * next_nodes + row] * deltas[0][row]; // effectively creating a memo recursive function since deltas is stored in class 
                        }                                                                           // could have kept it recursive by taking 'computeDeltas[row]' outside of for loops and storing in var                                                                                      
                        delta.push_back(tmp_val*layers[layer_num].activation.activationFunctionDerivative(layers[layer_num].getInputs()[col]));
                        tmp_val = 0;
                    }
                }
                deltas.insert(deltas.begin(),delta);
            }
        } 
        // recursive computeDeltas
        /*
        vector<double> computeDeltas(int layer_num) { 
            vector<double> delta;
            vector<double> next_delta; // since delta^i relies on delta^(i+1)
            double tmp_val = 0;

            if (layer_num == size(layers)-1){
                for (int i = 0; i < size(predictions); i++) {
                    delta.insert(delta.begin(),predictions[i]-actuals[i]);
                }
            } else {
                int prev_nodes = layers[layer_num].nodes;
                int next_nodes = layers[layer_num+1].nodes;
                next_delta = computeDeltas(layer_num + 1);

                for (int col = 0; col < prev_nodes; col++) { // transpose of weight matrix is used in calculation
                    for (int row = 0; row < next_nodes; row++){ // so row/col are backwards in loops
                        tmp_val += weights[layer_num+1][col*next_nodes + row] * next_delta[row]; 
                    }                                                                                                                                                              
                    delta.push_back(tmp_val*layers[layer_num].activation.pActivationDerivative(layers[layer_num].getInputs()[col]));
                    tmp_val = 0;
                }
            }
            deltas.insert(deltas.begin(),delta);
            return delta;
        } 
        */
        double calculateLoss(vector<double> preds, vector<double> acts) {
            return (*this.*pLoss)(preds,acts);
        }

        double predict(vector<double> input) { // forward propagation
            vector<double> tmp_output;
            layers[0].setInputs(input);
            for (int i = 0; i < size(layers) - 1; i++) {
                tmp_output = layers[i].computeOutput();
                layers[i+1].setInputs( weights[i].computeInput(tmp_output) );
            }
            predictions = layers[size(layers)-1].computeOutput();

            return calculateLoss(predictions,actuals);
        }

        void backPropagation(double learning_rate) {
            computeDeltas();
            for (int i = 0; i < size(weights); i++) {
                weights[i].backPropagationWeights(learning_rate, deltas[i], layers[i].getOutputs());
                weights[i].backPropagationBias(learning_rate, deltas[i]);
            }
        }

        void fit(vector<double> x_train, vector<double> y_train, int epochs, double lr, int num_feats) {
            input_data = x_train;
            num_features = num_feats;
            for (int i = 0; i < epochs; i++) {
                double error = 0.0;
                double avg_epoch_error;
                for (int j = 0; j < size(x_train)/num_feats; j++) {
                    vector<double> tmp_vec;
                    for (int k = 0; k < num_feats; k++) {
                        tmp_vec.push_back(x_train[j*num_feats + k]);
                    }
                    error += predict(tmp_vec);
                    backPropagation(lr);
                }
                avg_epoch_error = error / (size(x_train)/num_feats);
                cout << "Epoch " << i << '/' << epochs << "\tLoss = " << avg_epoch_error;
            }
        }

};


int main() {return 0;}