#include <iostream>
#include "util.h"
#include <torch/torch.h>

class CubeFunction : public torch::autograd::Function<CubeFunction>
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input)
    {
        ctx->save_for_backward({input});
        return input.pow(3);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        torch::Tensor input = saved[0];

        torch::Tensor grad_output = grad_outputs[0];

        // dy/dx = 3 x^2
        torch::Tensor grad_input = 3 * input.pow(2) * grad_output;

        return {grad_input};
    }
};

int main() {
    std::cout << "Hello, Terrain Building Layout!" << std::endl;
   
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random Tensor:\n" << tensor << std::endl;
    torch::Tensor x = torch::tensor({2.0}, torch::requires_grad());

    torch::Tensor y = CubeFunction::apply(x);
    y.backward();

    std::cout << "y = " << y << std::endl;            // 8
    std::cout << "dy/dx = " << x.grad() << std::endl; // 12
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    std::cout << "torch::cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "torch::cuda::device_count():" << torch::cuda::device_count() << std::endl;

    torch::Device device(torch::kCUDA);
    torch::Tensor tensor1 = torch::eye(3);         // (A) tensor-cpu
    torch::Tensor tensor2 = torch::eye(3, device); // (B) tensor-cuda
    std::cout << tensor1 << std::endl;
    std::cout << tensor2 << std::endl;
    return 0;
}