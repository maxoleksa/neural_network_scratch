#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Model {
    private:
        vector<vector<double>> outputs;
        vector<vector<double>> inputs;
        double y;
    public:
        double h;
        double learning_rate;
        vector<Layer> layers;
        vector<Weight> weights;
        vector<vector<double>> deltas;
        vector<string> activations;
        vector<vector<double>> biases;

        void compute_deltas(int layer) {
            double tmp;
            if (layer == size(layers)) {
                return outputs.back() - y;
            } else {
                for (int row = 0; row < size(weights[layer+1][row]); row++) {
                    for (int col = 0; col < size(weights[layer+1]); col++) {
                        tmp = weights[layer+1][col][row] * deltas[layer+1][row][col]
                    }
                }
            }
        }

        double forward_propagation(vector<double> input_data) {
            inputs.push_back(input_data);

            for (int layer = 0; layer < size(layers); layer++) {
                vector<double> tmp_vec;
                double tmp_val = 0.0;

                outputs.push_back(activation(inputs[layer]));

                for (int row = 0; row < size(weights[layer]); row++) {
                    for (int col = 0; col < size(weights[layer][row]); col++) {
                        tmp_val += weights[layer][row][col]*outputs[layer][col] + biases[layer][col];
                    }
                    tmp_vec.push_back(tmp_val);
                }

                inputs.push_back(tmp_vec);

                delete &tmp_vec;
                delete &tmp_val;
            }

            return outputs.back()
        }

        
};

int main() {
    return 0
}