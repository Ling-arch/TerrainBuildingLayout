#include "rvd.h"

namespace rvd{

    RectangularVoronoi::RectangularVoronoi(
        int N,
        int M,
        float temperature,
        float aspect_x,
        float aspect_y
    )
        : num_seeds(N),
          num_samples(M),
          tau(temperature),
          ax(aspect_x),
          ay(aspect_y)
    {
        seeds = torch::rand({N, 2}, torch::requires_grad());
        samples = torch::rand({M, 2});
    }

    // soft max(|dx|, |dy|)
    torch::Tensor RectangularVoronoi::soft_rect_dist(const torch::Tensor& dx,const torch::Tensor& dy) const{
        auto a = dx / ax;
        auto b = dy / ay;
        // log-sum-exp approx of max
        return tau * torch::log(torch::exp(a / tau) + torch::exp(b / tau));
    }

    // compute distance matrix [M,N]
    torch::Tensor RectangularVoronoi::compute_distances() const
    {
        auto x = samples.unsqueeze(1); // [M,1,2]
        auto s = seeds.unsqueeze(0);   // [1,N,2]

        auto diff = torch::abs(x - s); // [M,N,2]

        auto dx = diff.select(2, 0);
        auto dy = diff.select(2, 1);

        return soft_rect_dist(dx, dy); // [M,N]
    }

    // soft assignment p_i(x)
    torch::Tensor RectangularVoronoi::compute_weights() const
    {
        auto D = compute_distances();          // [M,N]
        auto W = torch::exp(-D / tau);         // [M,N]
        return W / W.sum(1, true);              // normalize
    }

    // Lloyd energy
    torch::Tensor RectangularVoronoi::lloyd_loss() const
    {
        auto P = compute_weights();             // [M,N]

        auto x = samples.unsqueeze(1);          // [M,1,2]
        auto s = seeds.unsqueeze(0);            // [1,N,2]

        auto diff = x - s;
        auto dist2 = diff.pow(2).sum(2);        // [M,N]

        return (P * dist2).sum();
    }
}