#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include "Parameters.h"

// Dynamically load dataset from the CSV in params.csvPath.
// Fills out featuresTensor + labelsTensor, and also returns a headerOut if needed.
void loadDataset(const Parameters &params,
                 torch::Tensor &featuresTensor,
                 torch::Tensor &labelsTensor,
                 std::vector<std::string> &headerOut);

#endif // DATASETLOADER_H
