#include <armadillo>
#include <random>
#include <vector>

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

private:
  Network(){};

  int layerNumber;
  std::vector<int> sizes;
  std::vector<arma::vec> biases;
  std::vector<arma::mat> weights;
};
