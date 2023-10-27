#ifndef MODEL_COMPONENT_CLASSES_H
#define MODEL_COMPONENT_CLASSES_H

class Activation {
    private:
        double (Activation::*pActivation) (double);
        double (Activation::*pActivationDerivative) (double);

        double sigmoid(double x);
        double sigmoid_prime(double x);
        double linear(double x);
        double linear_prime(double x);
        double relu(double x);
        double relu_prime(double x);
        double hyper_tan(double x);
        double hyper_tan_prime(double x);
    public:
        Activation(Activation &a);
        Activation();

        void operator=(const Activation &a);
        double activationFunction(double x);
        double activationFunctionDerivative(double x);
        vector<double> generateOutputs(vector<double> inputs);
};
class Weight {
    private:
    public:
        vector<double> weights;
        Layer prev_layer;
        Layer next_layer;
        vector<double> bias;

        Weight(Weight &w);
        Weight();
        Weight(Layer _prev, Layer _next);

        void operator=(const Weight &w);
        vector<double> computeInput(vector<double> output);
        void backPropagationWeights(double lr, vector<double> delta, vector<double> prev_outputs);
        void backPropagationBias(double lr, vector<double> delta);
};
class Layer {
    private:
        vector<double> inputs;
        vector<double> outputs;
    public:
        int nodes;
        Activation activation;

        Layer(Layer &l);
        Layer();
        Layer(int _nodes, string _activation);

        void operator=(const Layer &l);
        vector<double> computeOutput();
        vector<double> getInputs();
        vector<double> getOutputs();
};

#endif