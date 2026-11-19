// Host-side body storage, struct-of-arrays. Device backends mirror this layout.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

struct Bodies {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> m;
    // Stable identity: backends may reorder bodies (the tree methods sort them by Morton key),
    // so anything that tracks a particular body goes through `id`, never the array index.
    std::vector<uint32_t> id;

    std::size_t size() const { return x.size(); }

    void resize(std::size_t n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        m.resize(n); id.resize(n);
    }

    void reserve(std::size_t n) {
        x.reserve(n); y.reserve(n); z.reserve(n);
        vx.reserve(n); vy.reserve(n); vz.reserve(n);
        m.reserve(n); id.reserve(n);
    }

    void push(double px, double py, double pz, double pvx, double pvy, double pvz, double pm) {
        id.push_back(static_cast<uint32_t>(x.size()));
        x.push_back(float(px)); y.push_back(float(py)); z.push_back(float(pz));
        vx.push_back(float(pvx)); vy.push_back(float(pvy)); vz.push_back(float(pvz));
        m.push_back(float(pm));
    }
};
