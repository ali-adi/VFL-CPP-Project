#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>

struct Parameters {
    // Basic config
    std::string datasetArg; // e.g. "credit", "neurips-base", etc.
    std::string csvPath;    // Will store the CSV file path dynamically.

    // Training hyperparameters:
    int numEpochs;
    double learningRate;
    int64_t batchSize;

    // Ratios for train/val/test
    double trainRatio;
    double valRatio;
    double testRatio;
};

// Returns a default Parameters struct for the given datasetArg.
Parameters getParameters(const std::string &datasetArg);

#endif // PARAMETERS_H
