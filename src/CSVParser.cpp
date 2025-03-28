#include "CSVParser.h"
#include <fstream>
#include <sstream>

std::vector<std::vector<std::string>> loadCSV(const std::string &filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        return data; // empty if not found
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(token);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}
