#include <armadillo>
#include <cmath>
#include <vector>

typedef std::pair<arma::mat, int> trainingData;

class Network {
public:
  explicit Network(std::vector<int> sizes) : sizes(sizes) {
    layerNumber = sizes.size();

    for (int i = 1; i < sizes.size(); ++i) {
      biases.push_back(arma::vec(sizes[i], arma::fill::randn));
    }

    for (int i = 0; i < sizes.size() - 1; ++i) {
      weights.push_back(arma::mat(sizes[i], sizes[i + 1], arma::fill::randn));
    }
  }

  static void sigmoid(arma::vec &z) {
    z.transform([](double val) { return 1 / (1 + exp(-val)); });
  }

  arma::vec feedforward(const arma::vec &input);
  void sgd(std::vector<trainingData> &trainingSet, const int epochs,
           const int miniBatchSize, const double eta);
  void updateMiniBatch(std::vector<trainingData *> &miniBatch,
                       const double eta);

private:
  Network(){};

  int layerNumber;
  std::vector<int> sizes;
  // biases[i] represents bias at (i+1)th layer
  std::vector<arma::vec> biases;
  // weights[i] represents weight at ith layer
  std::vector<arma::mat> weights;
};
