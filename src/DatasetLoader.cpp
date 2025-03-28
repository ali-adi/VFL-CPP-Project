#include "DatasetLoader.h"
#include "CSVParser.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cctype>    // for std::isspace

// Helper to parse float or fallback to 0.0
static float safeStof(const std::string &s) {
    try {
        return std::stof(s);
    } catch (...) {
        return 0.0f;
    }
}

// Trim leading + trailing whitespace (including \r or spaces).
static void trim(std::string &str) {
    // Trim front
    while (!str.empty() && std::isspace((unsigned char)str.front())) {
        str.erase(str.begin());
    }
    // Trim back
    while (!str.empty() && std::isspace((unsigned char)str.back())) {
        str.pop_back();
    }
}

void loadDataset(const Parameters &params,
                 torch::Tensor &featuresTensor,
                 torch::Tensor &labelsTensor,
                 std::vector<std::string> &headerOut)
{
    auto csvData = loadCSV(params.csvPath);
    if (csvData.empty()) {
        std::cerr << "ERROR: CSV is empty or cannot open: " << params.csvPath << std::endl;
        return;
    }
    // 1) First row is header
    headerOut = csvData[0];

    // If credit or credit-balanced dataset might have an empty first token => skip row
    if ((params.datasetArg == "credit" || params.datasetArg == "credit-balanced")
        && !headerOut.empty() && headerOut[0].empty())
    {
        if (csvData.size() > 1) {
            headerOut = csvData[1];
            csvData.erase(csvData.begin(), csvData.begin() + 2);
        } else {
            std::cerr << "ERROR: Not enough lines after skipping row." << std::endl;
            return;
        }
    } else {
        // remove the header from data
        csvData.erase(csvData.begin());
    }

    // 2) Trim each header column to remove possible trailing or leading whitespace
    for (auto &col : headerOut) {
        trim(col);
    }

    // 3) Identify target column
    int targetIndex = -1;
    std::vector<int> featureIndices;
    for (int i = 0; i < (int)headerOut.size(); i++) {
        auto &col = headerOut[i];
        if (col == "ID") {
            continue;
        }
        // Accept either "default payment next month", "fraud_bool", or "Y"
        if (col == "default payment next month" ||
            col == "fraud_bool" ||
            col == "Y")
        {
            targetIndex = i;
            continue;
        }
        featureIndices.push_back(i);
    }
    if (targetIndex < 0) {
        std::cerr << "ERROR: No recognized target col in header." << std::endl;
        return;
    }

    // 4) Parse rows
    std::vector<std::vector<float>> allFeatures;
    std::vector<float> allLabels;

    for (auto &row : csvData) {
        // Also trim each token in data row to remove \r, etc.
        for (auto &token : row) {
            trim(token);
        }
        if ((int)row.size() < (int)headerOut.size()) {
            // skip incomplete row
            continue;
        }
        float lbl = safeStof(row[targetIndex]);
        allLabels.push_back(lbl);

        std::vector<float> feats;
        for (int idx : featureIndices) {
            feats.push_back(safeStof(row[idx]));
        }
        allFeatures.push_back(feats);
    }

    // 5) Shuffle
    {
        std::vector<size_t> indices(allFeatures.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<std::vector<float>> shufFeat;
        std::vector<float> shufLab;
        shufFeat.reserve(indices.size());
        shufLab.reserve(indices.size());
        for (auto idx : indices) {
            shufFeat.push_back(allFeatures[idx]);
            shufLab.push_back(allLabels[idx]);
        }
        allFeatures = std::move(shufFeat);
        allLabels   = std::move(shufLab);
    }

    // 6) Convert to Torch Tensors
    int64_t nRows = (int64_t)allFeatures.size();
    int64_t nCols = (int64_t)featureIndices.size();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    featuresTensor = torch::empty({nRows, nCols}, opts);

    {
        auto acc = featuresTensor.accessor<float,2>();
        for (int64_t i = 0; i < nRows; i++) {
            for (int64_t j = 0; j < nCols; j++) {
                acc[i][j] = allFeatures[i][j];
            }
        }
    }
    labelsTensor = torch::from_blob(allLabels.data(), {nRows, 1}, opts).clone();
}
