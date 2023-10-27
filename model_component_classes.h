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
        std::vector<double> generateOutputs(std::vector<double> inputs);
};

class Layer {
    private:
        std::vector<double> inputs;
        std::vector<double> outputs;
    public:
        int nodes;
        Activation activation;

        Layer(Layer &l);
        Layer();
        Layer(int _nodes, std::string _activation);

        void operator=(const Layer &l);
        std::vector<double> computeOutput();
        std::vector<double> getInputs();
        std::vector<double> getOutputs();
};

class Weight {
    private:
    public:
        std::vector<double> weights;
        Layer prev_layer;
        Layer next_layer;
        std::vector<double> bias;

        Weight(Weight &w);
        Weight();
        Weight(Layer _prev, Layer _next);

        void operator=(const Weight &w);
        std::vector<double> computeInput(std::vector<double> output);
        void backPropagationWeights(double lr, std::vector<double> delta, std::vector<double> prev_outputs);
        void backPropagationBias(double lr, std::vector<double> delta);
};


#endif