#ifndef MYDATASET_H
#define MYDATASET_H

#include <torch/torch.h>

// A simple custom dataset that returns an example with data + target
struct MyDataset : torch::data::datasets::Dataset<MyDataset> {
    torch::Tensor data_, targets_;

    MyDataset(const torch::Tensor &data, const torch::Tensor &targets)
        : data_(data), targets_(targets) {}

    torch::data::Example<> get(size_t index) override {
        return {data_[index], targets_[index]};
    }

    torch::optional<size_t> size() const override {
        return data_.size(0);
    }
};

#endif // MYDATASET_H
