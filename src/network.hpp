#include <armadillo>
#include <cmath>
#include <vector>

typedef std::pair<arma::vec, arma::vec> trainingData;

class Network {
public:
  explicit Network(std::vector<int> sizes) : sizes(sizes) {
    layerNumber = sizes.size();

    for (int i = 1; i < sizes.size(); ++i) {
      biases.push_back(arma::vec(sizes[i], arma::fill::randn));
      biasesShape.push_back(arma::vec(sizes[i], arma::fill::zeros));
    }

    for (int i = 0; i < sizes.size() - 1; ++i) {
      weights.push_back(arma::mat(sizes[i], sizes[i + 1], arma::fill::randn));
      weightsShape.push_back(
          arma::mat(sizes[i], sizes[i + 1], arma::fill::zeros));
    }
  }

  static arma::vec sigmoid(arma::vec z) {
    z.transform([](double val) { return 1 / (1 + exp(-val)); });
    return z;
  }

  static arma::vec sigmoidPrime(arma::vec z) {
    arma::vec sigmoidZ = sigmoid(z);
    arma::vec rhs = sigmoidZ;
    rhs.transform([](double val) { return 1 - val; });
    return sigmoidZ * arma::diagmat(rhs);
  }

  arma::vec feedforward(const arma::vec &input);
  void sgd(std::vector<trainingData> &trainingSet, const int epochs,
           const int miniBatchSize, const double eta);
  void updateMiniBatch(std::vector<trainingData *> &miniBatch,
                       const double eta);
  void backprop(arma::vec &in, arma::vec &out, std::vector<arma::vec> &partialB,
                std::vector<arma::mat> &partialW);
  arma::vec costDerivative(arma::vec &out, arma::vec &y);

private:
  Network(){};

  int layerNumber;
  std::vector<int> sizes;
  // biases[i] represents bias at (i+1)th layer
  std::vector<arma::vec> biases, biasesShape;
  // weights[i] represents weight from ith layer to (i+1)th layer
  std::vector<arma::mat> weights, weightsShape;
};
