// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

namespace owen {

class NNUECudaBackend {
public:
    NNUECudaBackend() = default;
    ~NNUECudaBackend();

    bool available() const;
    bool upload(int hidden, const std::vector<float>& w1, const std::vector<float>& b1,
                const std::vector<float>& w2, float b2, std::string* error);
    bool evaluate(const std::vector<int>& features, float* out, std::string* error) const;

private:
    static constexpr int MAX_FEATURES = 37;
    void release();

    void* d_w1_ = nullptr;
    void* d_b1_ = nullptr;
    void* d_w2_ = nullptr;
    void* d_features_ = nullptr;
    void* d_out_ = nullptr;
    float b2_ = 0.0f;
    int hidden_ = 0;
};

} // namespace owen
