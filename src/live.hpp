// Live view: a small HTTP server that streams the running simulation to a browser as MJPEG and
// takes camera and run controls back. POSIX sockets only, no dependencies.
//
// Threads: the simulation thread hands over body buffers (a swap, no copy) and blocks in gate()
// while paused; a render thread turns the newest buffer into a JPEG; a server thread accepts
// connections and one thread per connection serves it. The simulation never waits for rendering
// or the network: a frame is offered only when the renderer is idle, otherwise it is skipped.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bodies.hpp"
#include "render.hpp"

struct LiveOptions {
    int port = 8080;
    std::string bind = "127.0.0.1";
    int width = 960, height = 720;
    double fps = 30;
    int quality = 80;
    std::string view = "20,60";  // initial camera, same syntax as Config::view
    bool start_paused = false;
};

struct LiveStats {
    std::string scenario, backend, method;
    uint64_t n = 0;
    uint64_t step = 0, last_step = 0;  // last_step = 0: runs until stopped
    double time = 0;
    uint64_t diag_step = 0;            // step of the last diagnostic (de, virial)
    double de = 0, virial = 0;
    double ms_per_step = 0;
    bool finished = false;
};

class LiveView {
public:
    // Starts the threads; throws std::runtime_error if the address cannot be bound.
    LiveView(const LiveOptions& o, std::FILE* log);
    ~LiveView();
    LiveView(const LiveView&) = delete;
    LiveView& operator=(const LiveView&) = delete;

    std::string url() const;
    int port() const { return port_; }

    // Simulation side.
    bool frame_due();          // the renderer is idle and the frame interval has passed
    void submit(Bodies& b);    // swaps b with the view's spare buffer; b gets that spare back
    void set_stats(const LiveStats& s);
    // Blocks while paused. Returns false when Stop was pressed, true to take (at least) one step.
    bool gate();
    bool stop_requested();
    // After the last step: keeps serving until Stop is pressed.
    void hold();

private:
    void render_loop();
    void accept_loop();
    void serve(int fd);
    std::string status_json();
    std::string control(const std::string& query);

    LiveOptions opt_;
    std::FILE* log_;
    int listen_fd_ = -1;
    int port_ = 0;

    std::mutex mu_;
    std::condition_variable render_cv_, frame_cv_, control_cv_, conn_cv_;
    bool quit_ = false;

    // Bodies: the simulation swaps into pending_; the render thread swaps pending_ into working_.
    Bodies pending_, working_;
    bool have_pending_ = false, have_working_ = false, busy_ = false;
    std::chrono::steady_clock::time_point last_submit_{};

    // Camera, guarded by mu_.
    double yaw_ = 0, pitch_ = 0, zoom_ = 1;
    double base_extent_ = 0;  // 0 = refit on the next frame
    ColorMode color_ = ColorMode::density;
    int width_, height_;
    bool camera_dirty_ = false;

    // Latest frame.
    std::vector<unsigned char> jpeg_;
    uint64_t seq_ = 0;
    double render_ms_ = 0, fps_ = 0;
    std::chrono::steady_clock::time_point last_frame_{};

    // Run control and stats.
    bool paused_ = false, step_once_ = false, stop_ = false;
    LiveStats stats_;

    int clients_ = 0;      // open MJPEG streams
    int connections_ = 0;  // open connection threads; the destructor waits for 0

    std::thread render_thread_, accept_thread_;
};
