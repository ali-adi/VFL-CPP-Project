#ifndef TRAINUTILS_H
#define TRAINUTILS_H

#include <torch/torch.h>
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"
#include "EvaluateUtils.h"

struct TrainResult {
    double trainLoss;
    double valLoss;
    double valAccuracy;
    double best_val_accuracy;
};

// EXACTLY nine parameters for trainModel.
// Use operator() instead of .forward(...) for BCEWithLogitsLoss
template<typename LossModule, typename TrainLoader, typename ValLoader>
TrainResult trainModel(LRLocal lrModel,
                       MLP4Local mlpModel,
                       VFLAggregator agg,
                       TrainLoader &trainLoader,
                       ValLoader &valLoader,
                       LossModule &criterion,
                       torch::optim::Optimizer &optimizer,
                       int numEpochs,
                       int splitCol)
{
    TrainResult result {0.0, 0.0, 0.0, 0.0};
    double bestValAcc = 0.0;

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // TRAIN
        lrModel->train();
        mlpModel->train();
        agg->train();
        double runningLoss = 0.0;
        int totalSamples = 0;

        for (auto &batch : trainLoader) {
            auto inputs  = batch.data;
            auto targets = batch.target;
            int bsize    = inputs.size(0);
            totalSamples += bsize;

            auto left  = inputs.slice(/*dim=*/1, 0, splitCol);
            auto right = inputs.slice(/*dim=*/1, splitCol, inputs.size(1));
            auto out_left  = lrModel->forward(left);
            auto out_right = mlpModel->forward(right);
            auto final_out = agg->forward({out_left, out_right});

            optimizer.zero_grad();
            // Use criterion(...) instead of criterion.forward(...)
            auto loss = criterion(final_out, targets);
            loss.backward();
            optimizer.step();

            runningLoss += loss.template item<double>() * bsize;
        }
        double avgTrainLoss = totalSamples ? (runningLoss / totalSamples) : 0.0;

        // VALIDATION
        lrModel->eval();
        mlpModel->eval();
        agg->eval();
        double valLossSum = 0.0;
        int valSamples = 0;

        for (auto &batch : valLoader) {
            auto inputs  = batch.data;
            auto targets = batch.target;
            int bsize    = inputs.size(0);
            valSamples  += bsize;

            auto left  = inputs.slice(/*dim=*/1, 0, splitCol);
            auto right = inputs.slice(/*dim=*/1, splitCol, inputs.size(1));
            auto out_left  = lrModel->forward(left);
            auto out_right = mlpModel->forward(right);
            auto final_out = agg->forward({out_left, out_right});

            // operator() call
            auto loss = criterion(final_out, targets);
            valLossSum += loss.template item<double>() * bsize;
        }
        double avgValLoss = valSamples ? (valLossSum / valSamples) : 0.0;

        double valAcc = compute_accuracy(lrModel, mlpModel, agg, valLoader, splitCol);
        if (valAcc > bestValAcc) {
            bestValAcc = valAcc;
        }

        result.trainLoss         = avgTrainLoss;
        result.valLoss           = avgValLoss;
        result.valAccuracy       = valAcc;
        result.best_val_accuracy = bestValAcc;
    }

    return result;
}

#endif // TRAINUTILS_H
