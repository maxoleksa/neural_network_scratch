#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

#include "model_component_classes.h"

class Model{
    private:
        std::vector<Layer> layers;
        std::vector<Weight> weights;
        std::vector<std::vector<double>> deltas;

        int num_features;

        double (Model::*pLoss) (std::vector<double>, std::vector<double>);
        double logLoss(std::vector<double> preds, std::vector<double> acts);
        double mse(std::vector<double> preds, std::vector<double> acts);
    public:
        std::vector<double> predictions;
        std::vector<double> actuals;
        std::vector<double> input_data;
        double learning_rate;
        double loss;

        Model();

        void add(Layer _layer);
        void useLoss(std::string _loss);
        void computeDeltas();
        double calculateLoss(std::vector<double> preds, std::vector<double> acts);
        double predict(std::vector<double> input);
        void backPropagation(double learning_rate);
        void fit(std::vector<double> x_train, std::vector<double> y_train, int epochs, double lr, int num_feats);
};

#endif