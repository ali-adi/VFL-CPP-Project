#include "Parameters.h"
#include <iostream>

Parameters getParameters(const std::string &datasetArg) {
    Parameters p;
    p.datasetArg   = datasetArg;
    p.csvPath      = "";      // We’ll fill it in main if we want to.
    p.numEpochs    = 800;
    p.learningRate = 0.001;
    p.batchSize    = 128;

    // e.g. 70/15/15
    p.trainRatio = 0.7;
    p.valRatio   = 0.15;
    p.testRatio  = 0.15;

    return p;
}
