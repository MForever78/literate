#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "network.hpp"
#include <random>

using namespace std;
using namespace arma;

vec Network::feedforward(const vec &input) {
  vec output(input);
  for (int i = 0; i < sizes.size() - 1; ++i) {
    output = output * weights[i] + biases[i];
    sigmoid(output);
  }
  return output;
}

void Network::sgd(vector<trainingData> &trainingSet, const int epochs,
                  const int miniBatchSize, const double eta) {
  for (int i = 0; i < epochs; ++i) {
    // shuffle the traningData, takes O(n) time
    random_shuffle(trainingSet.begin(), trainingSet.end());

    // construct the miniBatches sets;
    vector<vector<trainingData *>> miniBatches;
    // only train batch whose size is equal to miniBatchSize
    // the remaining part would be dropped
    int fullMiniBatchSize = trainingSet.size() / miniBatchSize;
    for (int j = 0; j < fullMiniBatchSize; ++j) {
      vector<trainingData *> miniBatch;
      for (int k = 0; k < miniBatchSize; ++k) {
        miniBatch.push_back(&trainingSet[k]);
      }
      miniBatches.push_back(miniBatch);
    }

    for (auto &&miniBatch : miniBatches) {
      updateMiniBatch(miniBatch, eta);
    }
  }
}

void Network::updateMiniBatch(std::vector<trainingData *> &miniBatch,
                              const double eta) {}

#endif
