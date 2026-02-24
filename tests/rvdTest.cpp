#include "rvd.h"
#include <torch/optim/adam.h>
#include <iostream>

int main() {
    constexpr int N = 10;
    constexpr int M = 2048;

    rvd::RectangularVoronoi rvd(N, M, 0.05f);

    torch::optim::Adam optimizer(
        {rvd.seeds},
        torch::optim::AdamOptions(0.05)
    );

    for (int iter = 0; iter < 300; ++iter) {
        optimizer.zero_grad();

        auto loss = rvd.lloyd_loss();
        loss.backward();

        optimizer.step();

        if (iter % 20 == 0) {
            std::cout << "Iter " << iter
                      << " | Loss = " << loss.item<float>()
                      << std::endl;
        }
    }

    std::cout << "Final seeds:\n" << rvd.seeds << std::endl;
    return 0;
}