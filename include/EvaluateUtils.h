#ifndef EVALUATEUTILS_H
#define EVALUATEUTILS_H

#include <torch/torch.h>
#include "LocalModels.h"
#include "VFLAggregator.h"

// Template so it can accept the same loader type as training
template<typename Loader>
double compute_accuracy(LRLocal lrModel,
                        MLP4Local mlpModel,
                        VFLAggregator agg,
                        Loader &loader,
                        int splitCol)
{
    lrModel->eval();
    mlpModel->eval();
    agg->eval();

    size_t totalCorrect = 0;
    size_t totalSamples = 0;

    for (auto &batch : loader) {
        auto inputs  = batch.data;
        auto targets = batch.target;

        auto left  = inputs.slice(/*dim=*/1, 0, splitCol);
        auto right = inputs.slice(/*dim=*/1, splitCol, inputs.size(1));

        auto out_left  = lrModel->forward(left);
        auto out_right = mlpModel->forward(right);
        auto final_out = agg->forward({out_left, out_right});

        auto probs = torch::sigmoid(final_out);
        auto preds = (probs > 0.5).to(torch::kInt32);
        auto actual= targets.to(torch::kInt32);
        auto correctCount = preds.eq(actual).sum();

        totalCorrect += correctCount.template item<int64_t>();
        totalSamples += targets.size(0);
    }
    if (totalSamples == 0) return 0.0;
    return static_cast<double>(totalCorrect) / totalSamples;
}

#endif // EVALUATEUTILS_H
