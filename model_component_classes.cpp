#include <iostream>
#include <cmath>
#include <vector>

using namespace std;


class Weight {
    private:
        int next_nodes;
        int prev_nodes;
        vector<double> inputs;
        vector<double> outputs;
    public:
        double weights[next_nodes][prev_nodes];

        Weight(Layer prev_layer, Layer next_layer) {
            next_nodes = next_layer.num_nodes;
            prev_nodes = prev_layer.num_nodes;

            for (int i = 0; i < next_nodes; i++) {
                for (int j = 0; j < prev_nodes; j++) {
                    weights[i][j] = .1;
                }
            }
        }

        vector<double> generateOutputs(vector<double> inputs = inputs) {
            
        }
};

class Layer {
    private:
        double input[num_nodes];
        double output[num_nodes];
    public:
        int num_nodes;
        double bias[num_nodes];
        string activation;
        double learning_rate;

        // constructor 
        Layer(int _n, string _activation = "sigmoid", double _lr = .1) {
            num_nodes = _n;
            activation = _activation;
            learning_rate = _lr;
        }

        double forwardPropagation() {
            cout << "Not implemented.";
            throw; 
        }
        double backwardPropagation() {
            cout << "Not implemented.";
            throw;
        }
};

class Dense : public Layer {
    public:
        Dense(int _n, string _activation = "sigmoid", double _lr = .1) {
            num_nodes = _n;
            activation = _activation;
            learning_rate = _lr;
        }

        double forwardPropagation(double input[num_nodes]) { 
            for (int i = 0; i < num_nodes; i++) {
                output[i] = activation(input[i]);
            }
            return output;
        }

        double backwardPropagation(double delta[],double lr) {
            for (int i = 0; i < num_nodes; i++) {
                for (int j = 0; j <)
            }
        }
};


int main() {
    return 0
}