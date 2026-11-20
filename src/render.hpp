// Orthographic density render of the bodies: mass splatted into a float image, then a log tone
// map and a colour map. Pure CPU; used for PNG frames during a run, by lettuce-render, and by the
// live view.
#pragma once
#include <string>
#include <vector>

#include "bodies.hpp"

enum class ColorMode {
    density,  // brightness and colour from projected mass
    speed,    // brightness from projected mass, hue from mass-weighted mean speed (blue slow, red fast)
};

struct RenderOptions {
    int width = 1024;
    int height = 1024;
    std::string view = "xy";  // xy | xz | yz | <yaw_deg>,<pitch_deg>
    double extent = 0;        // half-width in world units; 0 = fit (99th percentile radius about the centre of mass, x1.2)
    ColorMode color = ColorMode::density;
};

struct Image {
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;  // width * height * 3, row 0 at the top
    double extent = 0;               // the half-width actually used (resolved when options.extent was 0)
};

// Throw std::runtime_error on a bad view string, bad size, or a write failure.
Image render_image(const Bodies& b, const RenderOptions& o);
void render_png(const Bodies& b, const RenderOptions& o, const std::string& path);
// Throw std::runtime_error unless `view` is a valid view string.
void validate_view(const std::string& view);
std::vector<unsigned char> encode_jpeg(const Image& img, int quality);
