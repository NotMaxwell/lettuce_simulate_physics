#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

#include "live.hpp"
#include "test.hpp"

namespace {

// One HTTP/1.0-style request; returns status line + headers + body.
std::string http_get(int port, const std::string& path) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) {
        ::close(fd);
        return "";
    }
    std::string req = "GET " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), 0);
    std::string out;
    char buf[65536];
    ssize_t k;
    while ((k = ::recv(fd, buf, sizeof buf, 0)) > 0) out.append(buf, static_cast<size_t>(k));
    ::close(fd);
    return out;
}

std::string body_of(const std::string& r) {
    size_t p = r.find("\r\n\r\n");
    return p == std::string::npos ? "" : r.substr(p + 4);
}

Bodies little_cluster() {
    Bodies b;
    for (int i = 0; i < 500; ++i) {
        double t = i * 0.37;
        b.push(std::cos(t) * (1 + 0.001 * i), std::sin(t) * (1 + 0.001 * i), 0.01 * (i % 7), -std::sin(t), std::cos(t), 0,
               1.0 / 500);
    }
    return b;
}

}  // namespace

TEST("live: serves page, status and frames; controls reach the simulation gate") {
    LiveOptions o;
    o.port = 0;
    o.width = 160;
    o.height = 120;
    std::FILE* log = std::fopen("/dev/null", "w");
    LiveView v(o, log);
    const int port = v.port();
    CHECK(port > 0);

    CHECK(http_get(port, "/").find("200 OK") != std::string::npos);
    CHECK(http_get(port, "/nope").find("404") != std::string::npos);

    LiveStats s;
    s.scenario = "test";
    s.n = 500;
    s.step = 7;
    v.set_stats(s);
    Bodies b = little_cluster();
    v.submit(b);

    std::string frame;
    for (int i = 0; i < 100 && frame.empty(); ++i) {
        std::string r = http_get(port, "/frame.jpg");
        if (r.find("200 OK") != std::string::npos) frame = body_of(r);
        else std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    CHECK(frame.size() > 100);
    CHECK(frame.size() > 2 && static_cast<unsigned char>(frame[0]) == 0xFF && static_cast<unsigned char>(frame[1]) == 0xD8);

    std::string st = body_of(http_get(port, "/status"));
    CHECK(st.find("\"scenario\":\"test\"") != std::string::npos);
    CHECK(st.find("\"step\":7") != std::string::npos);

    // Camera changes re-render the last bodies even without a new submit.
    st = body_of(http_get(port, "/control?yaw=45&pitch=30&color=speed&size=200x100"));
    CHECK(st.find("\"color\":\"speed\"") != std::string::npos);
    CHECK(st.find("\"width\":200") != std::string::npos);

    // Pause blocks the gate; Step releases it once; Stop makes it return false.
    http_get(port, "/control?pause=1");
    std::atomic<int> phase{0};
    std::thread sim([&] {
        bool a = v.gate();  // released by step
        phase = a ? 1 : -1;
        bool c = v.gate();  // released by stop
        phase = c ? -2 : 2;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(phase == 0);
    http_get(port, "/control?step=1");
    for (int i = 0; i < 100 && phase == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(phase == 1);
    http_get(port, "/control?stop=1");
    sim.join();
    CHECK(phase == 2);
    CHECK(v.stop_requested());
    std::fclose(log);
}

TEST("live: a busy port is reported, not ignored") {
    LiveOptions o;
    o.port = 0;
    std::FILE* log = std::fopen("/dev/null", "w");
    LiveView a(o, log);
    o.port = a.port();
    bool threw = false;
    try {
        LiveView b(o, log);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::fclose(log);
}

TEST("live: non-finite stats stay valid JSON") {
    LiveOptions o;
    o.port = 0;
    std::FILE* log = std::fopen("/dev/null", "w");
    LiveView v(o, log);
    LiveStats s;
    s.de = std::numeric_limits<double>::quiet_NaN();
    s.virial = std::numeric_limits<double>::infinity();
    v.set_stats(s);
    std::string st = body_of(http_get(v.port(), "/status"));
    CHECK(st.find("\"de\":null") != std::string::npos);
    CHECK(st.find("\"virial\":null") != std::string::npos);
    CHECK(st.find("nan") == std::string::npos && st.find("inf") == std::string::npos);
    std::fclose(log);
}

TEST("live: a stalled stream client does not block shutdown") {
    std::FILE* log = std::fopen("/dev/null", "w");
    int fd;
    auto t0 = std::chrono::steady_clock::now();
    {
        LiveOptions o;
        o.port = 0;
        o.width = 1024;
        o.height = 1024;
        LiveView v(o, log);
        Bodies b = little_cluster();
        v.submit(b);
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        int rcv = 1024;
        ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcv, sizeof rcv);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(static_cast<uint16_t>(v.port()));
        ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        CHECK(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
        const char req[] = "GET /stream HTTP/1.1\r\nHost: x\r\n\r\n";
        ::send(fd, req, sizeof req - 1, 0);
        // Never read: frames fill the socket buffers and the server's send stalls.
        for (int i = 0; i < 20; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            v.submit(b = little_cluster());
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ::close(fd);
    CHECK(secs < 8.0);
    std::fclose(log);
}
