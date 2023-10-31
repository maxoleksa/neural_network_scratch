#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

#include "model_component_classes.h"

class Model{
    private:
        

        int num_features;

        double (Model::*pLoss) (std::vector<double>, std::vector<double>);
        double binaryCrossEntropy(std::vector<double> preds, std::vector<double> acts);
        double logLoss(std::vector<double> preds, std::vector<double> acts);
        double mse(std::vector<double> preds, std::vector<double> acts);
        double mae(std::vector<double> preds, std::vector<double> acts);
    public:
        std::vector<double> predictions;
        std::vector<double> actuals;
        std::vector<double> input_data;
        double learning_rate;
        double loss;

        std::vector<Layer> layers;
        std::vector<Weight> weights;
        std::vector<std::vector<double>> deltas;

        Model();
        //Model(Model &m);


        void add(Layer _layer);
        void useLoss(std::string _loss);
        void computeDeltas(double actual);
        double calculateLoss(std::vector<double> preds, std::vector<double> acts);
        double predict(std::vector<double> input, double actual);
        void backPropagation(double learning_rate, double actual);
        void fit(std::vector<double> x_train, std::vector<double> y_train, int epochs, double lr, int num_feats);
};

#endif