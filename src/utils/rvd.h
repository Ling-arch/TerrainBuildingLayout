#include <torch/torch.h>
#include <vector>
#include <iostream>

namespace rvd
{

    class RectangularVoronoi
    {
    public:
        int num_seeds;
        int num_samples;

        float tau;    // soft temperature
        float ax, ay; // rectangular aspect

        torch::Tensor seeds;   // [N,2] requires_grad
        torch::Tensor samples; // [M,2]

        RectangularVoronoi(int N, int M, float temperature = 0.05f, float aspect_x = 1.f, float aspect_y = 1.f);

        // soft max(|dx|, |dy|)
        torch::Tensor soft_rect_dist(const torch::Tensor &dx, const torch::Tensor &dy) const;

        // compute distance matrix [M,N]
        torch::Tensor compute_distances() const;

        // soft assignment p_i(x)
        torch::Tensor compute_weights() const;

        // Lloyd energy
        torch::Tensor lloyd_loss() const;
    };

}