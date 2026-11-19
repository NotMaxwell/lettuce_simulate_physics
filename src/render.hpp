// Orthographic density render of the bodies to a PNG: mass splatted into a float image, then a
// log tone map and a fixed colour map. Pure CPU; used for frames during a run and by lettuce-render.
#pragma once
#include <string>

#include "bodies.hpp"

struct RenderOptions {
    int width = 1024;
    int height = 1024;
    std::string view = "xy";  // xy | xz | yz | <yaw_deg>,<pitch_deg>
    double extent = 0;        // half-width in world units; 0 = fit (99th percentile radius about the centre of mass, x1.2)
};

// Throws std::runtime_error on a bad view string or a write failure.
void render_png(const Bodies& b, const RenderOptions& o, const std::string& path);
