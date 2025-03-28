#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <random>

#include "Parameters.h"       // getParameters
#include "DatasetLoader.h"    // loadDataset
#include "TrainUtils.h"       // trainModel
#include "EvaluateUtils.h"    // compute_accuracy
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"

// Tokenize a line by comma
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    // 1) Parse dataset argument or default
    std::string datasetArg = "credit";
    if (argc > 1) {
        datasetArg = argv[1];
    }
    std::cout << "Selected dataset: " << datasetArg << std::endl;

    // 2) Get base parameters from getParameters
    Parameters params = getParameters(datasetArg);

    // 3) Dynamically set CSV path
    if (datasetArg == "credit") {
        params.csvPath = "../data/default_of_credit_card_clients.csv";
    } else if (datasetArg == "credit-balanced") {
        params.csvPath = "../data/default_of_credit_card_clients-balanced.csv";
    } else if (datasetArg == "neurips-base") {
        params.csvPath = "../data/Base.csv";
    } else if (datasetArg == "neurips-vari") {
        params.csvPath = "../data/Variant I.csv";
    } else if (datasetArg == "neurips-varii") {
        params.csvPath = "../data/Variant II.csv";
    } else if (datasetArg == "neurips-variii") {
        params.csvPath = "../data/Variant III.csv";
    } else if (datasetArg == "neurips-variv") {
        params.csvPath = "../data/Variant IV.csv";
    } else if (datasetArg == "neurips-varv") {
        params.csvPath = "../data/Variant V.csv";
    } else {
        std::cerr << "Unknown dataset: " << datasetArg << ". Defaulting to credit.\n";
        params.csvPath = "../data/default_of_credit_card_clients.csv";
    }

    // 4) Load dataset => produce features/labels
    torch::Tensor featuresTensor, labelsTensor;
    std::vector<std::string> header;
    loadDataset(params, featuresTensor, labelsTensor, header);

    int64_t numRows = featuresTensor.size(0);
    if (numRows == 0) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return -1;
    }
    int64_t numCols = featuresTensor.size(1);
    std::cout << "Loaded " << numRows << " rows, " << numCols << " features.\n";

    // 5) Split into train/val/test
    int64_t trainSize = static_cast<int64_t>(params.trainRatio * numRows);
    int64_t valSize   = static_cast<int64_t>(params.valRatio   * numRows);
    int64_t testSize  = numRows - trainSize - valSize;

    auto trainFeatures = featuresTensor.slice(0, 0, trainSize);
    auto trainLabels   = labelsTensor.slice(0, 0, trainSize);
    auto valFeatures   = featuresTensor.slice(0, trainSize, trainSize + valSize);
    auto valLabels     = labelsTensor.slice(0, trainSize, trainSize + valSize);
    auto testFeatures  = featuresTensor.slice(0, trainSize + valSize, numRows);
    auto testLabels    = labelsTensor.slice(0, trainSize + valSize, numRows);

    // 6) Create dataset/dataloaders
    auto trainDataset = MyDataset(trainFeatures, trainLabels).map(torch::data::transforms::Stack<>());
    auto valDataset   = MyDataset(valFeatures,   valLabels).map(torch::data::transforms::Stack<>());
    auto testDataset  = MyDataset(testFeatures,  testLabels).map(torch::data::transforms::Stack<>());

    auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(trainDataset), params.batchSize);
    auto valLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(valDataset), params.batchSize);
    auto testLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(testDataset), params.batchSize);

    // 7) model setup
    int64_t splitCol = numCols / 2;
    LRLocal lrLocalModel(splitCol, 64);
    MLP4Local mlpLocalModel(numCols - splitCol, 64);
    VFLAggregator aggregator(64 + 64, 1);

    lrLocalModel->train();
    mlpLocalModel->train();
    aggregator->train();

    // 8) BCEWithLogitsLoss as an object
    torch::nn::BCEWithLogitsLoss criterion;

    // 9) Construct Adam with param groups inline
    torch::optim::Adam optimizer(
        std::vector<torch::optim::OptimizerParamGroup>{
            torch::optim::OptimizerParamGroup(lrLocalModel->parameters()),
            torch::optim::OptimizerParamGroup(mlpLocalModel->parameters()),
            torch::optim::OptimizerParamGroup(aggregator->parameters())
        },
        torch::optim::AdamOptions(params.learningRate)
    );

    // Create a run folder
    std::filesystem::create_directories("../runs");

    // 10) Train
    TrainResult result = trainModel<
            torch::nn::BCEWithLogitsLoss,
            decltype(*trainLoader),
            decltype(*valLoader)
        >(
        lrLocalModel,
        mlpLocalModel,
        aggregator,
        *trainLoader,
        *valLoader,
        criterion,
        optimizer,
        params.numEpochs,
        splitCol
    );

    std::cout << "\nTraining complete.\n"
              << "Best val accuracy: " << (result.best_val_accuracy * 100.0) << "%\n"
              << "Final train loss: " << result.trainLoss << "\n"
              << "Final val accuracy: " << (result.valAccuracy * 100.0) << "%\n";

    // optional: test accuracy
    double testAcc = compute_accuracy(lrLocalModel, mlpLocalModel, aggregator, *testLoader, splitCol);
    std::cout << "Test accuracy: " << (testAcc * 100.0) << "%\n";

    return 0;
}
