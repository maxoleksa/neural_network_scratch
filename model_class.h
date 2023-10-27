#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

class Model{
    private:
        vector<Layer> layers;
        vector<Weight> weights;
        vector<vector<double>> deltas;

        int num_features;

        double (Model::*pLoss) (vector<double>, vector<double>);
        double logLoss(vector<double> preds, vector<double> acts);
        double mse(vector<double> preds, vector<double> acts);
    public:
        vector<double> predictions;
        vector<double> actuals;
        vector<double> input_data;
        double learning_rate;
        double loss;

        Model();

        void add(Layer _layer);
        void useLoss(string _loss);
        void computeDeltas();
        double calculateLoss(vector<double> preds, vector<double> acts);
        double predict(vector<double> input);
        void backPropagation(double learning_rate);
        void fit(vector<double> x_train, vector<double> y_train, int epochs, double lr, int num_feats);
};

#endif