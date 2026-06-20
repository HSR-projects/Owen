// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// nnue.h — tiny native evaluator loading weights exported by training/nnue.py.

#pragma once

#include "board.h"
#include <memory>
#include <string>
#include <vector>

namespace owen {

class NNUECudaBackend;

class NNUE {
public:
    NNUE();
    ~NNUE();

    bool load(const std::string& path, std::string* error = nullptr);
    bool loaded() const { return loaded_; }
    int evaluate(const Board& b) const;
    bool set_gpu_enabled(bool enabled, std::string* error = nullptr);
    bool gpu_enabled() const { return gpu_enabled_; }
    bool gpu_available() const;
    const char* backend_name() const;

private:
    static constexpr int INPUT_SIZE = 12 * 64 + 5;

    std::vector<float> w1_;
    std::vector<float> b1_;
    std::vector<float> w2_;
    float b2_ = 0.0f;
    int hidden_ = 0;
    bool loaded_ = false;
    bool sf_active_ = false;     // true when a HalfKP net is loaded (sfnnue)
    bool gpu_enabled_ = false;
    std::unique_ptr<NNUECudaBackend> cuda_;
};

int material_evaluate(const Board& b);

// Fast tapered material + piece-square evaluation only (no mobility/king-safety).
// Used at interior nodes for pruning decisions to keep the search fast.
int lazy_evaluate(const Board& b);

} // namespace owen
