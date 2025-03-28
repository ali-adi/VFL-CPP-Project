#ifndef VFLAGGREGATOR_H
#define VFLAGGREGATOR_H

#include <torch/torch.h>

// Aggregator that simply concatenates the local outputs and does a linear
struct VFLAggregatorImpl : torch::nn::Module {
    torch::nn::Linear linear{nullptr};

    VFLAggregatorImpl(int64_t inFeatures, int64_t outFeatures) {
        linear = register_module("linear", torch::nn::Linear(inFeatures, outFeatures));
    }

    // aggregator input is a list of Tensors that we cat along dim=1
    torch::Tensor forward(std::initializer_list<torch::Tensor> inputs) {
        auto catTensor = torch::cat(inputs, 1);
        return linear->forward(catTensor);
    }
};
TORCH_MODULE(VFLAggregator);

#endif // VFLAGGREGATOR_H
