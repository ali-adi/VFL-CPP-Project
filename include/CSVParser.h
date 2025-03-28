#ifndef CSVPARSER_H
#define CSVPARSER_H

#include <vector>
#include <string>

// Loads entire CSV into a vector of vector<string>.
std::vector<std::vector<std::string>> loadCSV(const std::string &filename);

#endif // CSVPARSER_H
