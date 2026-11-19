// Linked instead of cuda_backend.cu when the build has no CUDA compiler.
#include "backend.hpp"

std::unique_ptr<Backend> make_cuda_backend(int, std::string& why) {
    why = "this build has no CUDA support (configure with a CUDA compiler, LETTUCE_CUDA=ON)";
    return nullptr;
}

bool cuda_compiled() { return false; }

int cuda_device_count() { return 0; }
