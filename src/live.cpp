#include "live.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

namespace {

using Clock = std::chrono::steady_clock;

extern const char* const kPage;

// Per-connection-thread shutdown probe, so send_all need not be threaded through every helper.
thread_local const std::function<bool()>* t_quit = nullptr;

constexpr int kSendStallMs = 10000;

// A stalled peer must neither block shutdown nor pin a thread forever: wait in short polls, give up
// when the server quits or the socket makes no progress for kSendStallMs.
bool send_all(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    int stalled_ms = 0;
    while (len > 0) {
        ssize_t k = ::send(fd, p, len, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (k > 0) {
            p += k;
            len -= static_cast<size_t>(k);
            stalled_ms = 0;
            continue;
        }
        if (k < 0 && errno == EINTR) continue;
        if (k == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) return false;
        if ((t_quit && (*t_quit)()) || stalled_ms >= kSendStallMs) return false;
        pollfd pf{fd, POLLOUT, 0};
        if (::poll(&pf, 1, 100) == 0) stalled_ms += 100;
    }
    return true;
}

bool send_str(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

bool respond(int fd, const char* status, const char* type, const std::string& body, bool keep_alive) {
    std::string h = std::string("HTTP/1.1 ") + status + "\r\nContent-Type: " + type +
                    "\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nCache-Control: no-store\r\nConnection: " + (keep_alive ? "keep-alive" : "close") + "\r\n\r\n";
    return send_str(fd, h) && send_str(fd, body);
}

std::map<std::string, std::string> parse_query(const std::string& q) {
    std::map<std::string, std::string> out;
    size_t i = 0;
    while (i < q.size()) {
        size_t amp = q.find('&', i);
        if (amp == std::string::npos) amp = q.size();
        std::string kv = q.substr(i, amp - i);
        size_t eq = kv.find('=');
        if (eq != std::string::npos) out[kv.substr(0, eq)] = kv.substr(eq + 1);
        else if (!kv.empty()) out[kv] = "";
        i = amp + 1;
    }
    return out;
}

bool parse_num(const std::string& s, double& v) {
    char* end = nullptr;
    double x = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != 0 || !std::isfinite(x)) return false;
    v = x;
    return true;
}

// JSON has no nan/inf; the page treats null as "no value".
std::string json_num(double v, const char* fmt) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, fmt, v);
    return b;
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (static_cast<unsigned char>(c) >= 0x20) o += c;
    }
    return o + "\"";
}

// Camera angles for the fixed views, matching render.cpp's yaw-then-pitch projection.
void view_angles(const std::string& view, double& yaw, double& pitch) {
    if (view == "xy") { yaw = 0; pitch = 0; return; }
    if (view == "xz") { yaw = 0; pitch = -90; return; }
    if (view == "yz") { yaw = -90; pitch = -90; return; }
    size_t comma = view.find(',');
    double a = 0, b = 0;
    if (comma != std::string::npos && parse_num(view.substr(0, comma), a) && parse_num(view.substr(comma + 1), b)) {
        yaw = a;
        pitch = b;
        return;
    }
    throw std::runtime_error("live: bad view '" + view + "'");
}

}  // namespace

LiveView::LiveView(const LiveOptions& o, std::FILE* log)
    : opt_(o), log_(log), width_(o.width), height_(o.height) {
    view_angles(o.view, yaw_, pitch_);
    paused_ = o.start_paused;
    // A browser closing a stream mid-frame must not kill the simulation.
    std::signal(SIGPIPE, SIG_IGN);

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error(std::string("live: socket: ") + std::strerror(errno));
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(o.port));
    std::string host = o.bind == "localhost" ? "127.0.0.1" : o.bind;
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(listen_fd_);
        throw std::runtime_error("live: live_bind must be an IPv4 address, got '" + o.bind + "'");
    }
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || ::listen(listen_fd_, 16) < 0) {
        std::string why = std::strerror(errno);
        ::close(listen_fd_);
        throw std::runtime_error("live: cannot listen on " + o.bind + ":" + std::to_string(o.port) + ": " + why +
                                 " (pick another port with live=<port>)");
    }
    // Port 0 asks the kernel for a free one (used by the tests).
    socklen_t len = sizeof addr;
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    render_thread_ = std::thread(&LiveView::render_loop, this);
    accept_thread_ = std::thread(&LiveView::accept_loop, this);
}

LiveView::~LiveView() {
    {
        std::lock_guard lk(mu_);
        quit_ = true;
        stop_ = true;
    }
    render_cv_.notify_all();
    frame_cv_.notify_all();
    control_cv_.notify_all();
    if (accept_thread_.joinable()) accept_thread_.join();
    if (render_thread_.joinable()) render_thread_.join();
    std::unique_lock lk(mu_);
    // Connection threads are detached; each wakes within its poll timeout once quit_ is set.
    conn_cv_.wait(lk, [&] { return connections_ == 0; });
    lk.unlock();
    if (listen_fd_ >= 0) ::close(listen_fd_);
}

std::string LiveView::url() const { return "http://" + opt_.bind + ":" + std::to_string(port_) + "/"; }

bool LiveView::frame_due() {
    std::lock_guard lk(mu_);
    if (busy_ || have_pending_) return false;
    // Nobody watching: keep a fresh frame at 1 fps so a new browser sees something at once.
    double interval = clients_ > 0 ? 1.0 / opt_.fps : 1.0;
    return std::chrono::duration<double>(Clock::now() - last_submit_).count() >= interval;
}

void LiveView::submit(Bodies& b) {
    {
        std::lock_guard lk(mu_);
        std::swap(pending_, b);
        have_pending_ = true;
        last_submit_ = Clock::now();
    }
    render_cv_.notify_one();
}

void LiveView::set_stats(const LiveStats& s) {
    std::lock_guard lk(mu_);
    stats_ = s;
}

bool LiveView::gate() {
    std::unique_lock lk(mu_);
    control_cv_.wait(lk, [&] { return stop_ || !paused_ || step_once_; });
    if (stop_) return false;
    step_once_ = false;
    return true;
}

bool LiveView::stop_requested() {
    std::lock_guard lk(mu_);
    return stop_;
}

void LiveView::hold() {
    std::unique_lock lk(mu_);
    stats_.finished = true;
    control_cv_.wait(lk, [&] { return stop_; });
}

void LiveView::render_loop() {
    std::unique_lock lk(mu_);
    while (true) {
        render_cv_.wait(lk, [&] { return quit_ || have_pending_ || (camera_dirty_ && have_working_); });
        if (quit_) return;
        if (have_pending_) {
            std::swap(working_, pending_);
            have_pending_ = false;
            have_working_ = true;
        }
        camera_dirty_ = false;
        RenderOptions ro;
        ro.width = width_;
        ro.height = height_;
        ro.view = std::to_string(yaw_) + "," + std::to_string(pitch_);
        ro.extent = base_extent_ > 0 ? base_extent_ / zoom_ : 0;
        const double used_zoom = zoom_;
        const uint64_t used_gen = fit_gen_;
        ro.color = color_;
        const int quality = opt_.quality;
        busy_ = true;
        lk.unlock();

        auto t0 = Clock::now();
        std::vector<unsigned char> jpeg;
        double fitted = 0;
        try {
            Image img = render_image(working_, ro);
            fitted = img.extent;
            jpeg = encode_jpeg(img, quality);
        } catch (const std::exception& e) {
            std::fprintf(log_, "# live: render failed: %s\n", e.what());
        }
        auto t1 = Clock::now();

        lk.lock();
        busy_ = false;
        if (jpeg.empty()) continue;
        // An auto-fitted frame fixes the extent, so zooming and rotating do not rescale on their own.
        // The fitted extent is for the zoom it was rendered at; a "fit" request meanwhile voids it.
        if (ro.extent == 0 && used_gen == fit_gen_) base_extent_ = fitted * used_zoom;
        jpeg_ = std::move(jpeg);
        ++seq_;
        render_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double dt = std::chrono::duration<double>(t1 - last_frame_).count();
        if (dt > 0 && dt < 10) fps_ = fps_ == 0 ? 1 / dt : 0.8 * fps_ + 0.2 / dt;
        last_frame_ = t1;
        frame_cv_.notify_all();
    }
}

void LiveView::accept_loop() {
    while (true) {
        {
            std::lock_guard lk(mu_);
            if (quit_) return;
        }
        pollfd p{listen_fd_, POLLIN, 0};
        int r = ::poll(&p, 1, 200);
        if (r <= 0) continue;
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        {
            std::lock_guard lk(mu_);
            ++connections_;
        }
        std::thread([this, fd] {
            serve(fd);
            ::close(fd);
            std::lock_guard lk(mu_);
            --connections_;
            conn_cv_.notify_all();
        }).detach();
    }
}

void LiveView::serve(int fd) {
    const std::function<bool()> quit = [this] {
        std::lock_guard lk(mu_);
        return quit_;
    };
    t_quit = &quit;
    struct Reset { ~Reset() { t_quit = nullptr; } } reset;
    std::string buf;
    char chunk[4096];
    while (true) {
        // Read one request head; idle keep-alive connections are dropped after 60 s.
        size_t end;
        int idle_ms = 0;
        while ((end = buf.find("\r\n\r\n")) == std::string::npos) {
            {
                std::lock_guard lk(mu_);
                if (quit_) return;
            }
            if (buf.size() > 16384 || idle_ms > 60000) return;
            pollfd p{fd, POLLIN, 0};
            int r = ::poll(&p, 1, 200);
            if (r == 0) { idle_ms += 200; continue; }
            if (r < 0) return;
            ssize_t k = ::recv(fd, chunk, sizeof chunk, 0);
            if (k <= 0) return;
            buf.append(chunk, static_cast<size_t>(k));
        }
        std::string head = buf.substr(0, end);
        buf.erase(0, end + 4);

        size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) return;
        std::string method = head.substr(0, sp1), target = head.substr(sp1 + 1, sp2 - sp1 - 1);
        std::string lower = head;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        bool keep = lower.find("connection: close") == std::string::npos;
        if (method != "GET") {
            respond(fd, "405 Method Not Allowed", "text/plain", "GET only\n", false);
            return;
        }
        size_t qm = target.find('?');
        std::string path = target.substr(0, qm), query = qm == std::string::npos ? "" : target.substr(qm + 1);

        if (path == "/" || path == "/index.html") {
            if (!respond(fd, "200 OK", "text/html; charset=utf-8", kPage, keep) || !keep) return;
        } else if (path == "/status") {
            if (!respond(fd, "200 OK", "application/json", status_json(), keep) || !keep) return;
        } else if (path == "/control") {
            if (!respond(fd, "200 OK", "application/json", control(query), keep) || !keep) return;
        } else if (path == "/frame.jpg") {
            std::vector<unsigned char> f;
            {
                std::lock_guard lk(mu_);
                f = jpeg_;
            }
            if (f.empty()) {
                if (!respond(fd, "503 Service Unavailable", "text/plain", "no frame yet\n", keep) || !keep) return;
            } else if (!respond(fd, "200 OK", "image/jpeg", std::string(f.begin(), f.end()), keep) || !keep) {
                return;
            }
        } else if (path == "/stream") {
            if (!send_str(fd, "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=lettuceframe\r\n"
                              "Cache-Control: no-store\r\nConnection: close\r\n\r\n"))
                return;
            uint64_t sent = 0;
            {
                std::lock_guard lk(mu_);
                ++clients_;
            }
            std::vector<unsigned char> f;
            while (true) {
                {
                    std::unique_lock lk(mu_);
                    frame_cv_.wait_for(lk, std::chrono::milliseconds(500), [&] { return quit_ || seq_ > sent; });
                    if (quit_) break;
                    if (seq_ <= sent) continue;
                    // Always the newest frame: a slow link skips frames instead of queueing them.
                    f = jpeg_;
                    sent = seq_;
                }
                std::string part = "--lettuceframe\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                   std::to_string(f.size()) + "\r\n\r\n";
                if (!send_str(fd, part) || !send_all(fd, f.data(), f.size()) || !send_str(fd, "\r\n")) break;
            }
            std::lock_guard lk(mu_);
            --clients_;
            return;
        } else {
            if (!respond(fd, "404 Not Found", "text/plain", "not found\n", keep) || !keep) return;
        }
    }
}

std::string LiveView::status_json() {
    std::lock_guard lk(mu_);
    const LiveStats& s = stats_;
    auto u = [](uint64_t v) { return std::to_string(v); };
    auto bl = [](bool v) { return v ? "true" : "false"; };
    const std::string num =
        "\"n\":" + u(s.n) + ",\"step\":" + u(s.step) + ",\"last_step\":" + u(s.last_step) +
        ",\"time\":" + json_num(s.time, "%.9g") + ",\"diag_step\":" + u(s.diag_step) +
        ",\"de\":" + json_num(s.de, "%.6g") + ",\"virial\":" + json_num(s.virial, "%.6g") +
        ",\"ms_per_step\":" + json_num(s.ms_per_step, "%.6g") + ",\"finished\":" + bl(s.finished) +
        ",\"paused\":" + bl(paused_) + ",\"stopped\":" + bl(stop_) + ",\"yaw\":" + json_num(yaw_, "%.6g") +
        ",\"pitch\":" + json_num(pitch_, "%.6g") + ",\"zoom\":" + json_num(zoom_, "%.6g") +
        ",\"width\":" + std::to_string(width_) + ",\"height\":" + std::to_string(height_) +
        ",\"fps\":" + json_num(fps_, "%.3g") + ",\"render_ms\":" + json_num(render_ms_, "%.4g") +
        ",\"clients\":" + std::to_string(clients_) + ",\"frame\":" + u(seq_);
    return "{\"scenario\":" + json_str(s.scenario) + ",\"backend\":" + json_str(s.backend) +
           ",\"method\":" + json_str(s.method) + ",\"color\":" +
           json_str(color_ == ColorMode::speed ? "speed" : "density") + "," + num + "}";
}

std::string LiveView::control(const std::string& query) {
    auto q = parse_query(query);
    bool camera = false, run = false;
    {
        std::lock_guard lk(mu_);
        double v = 0;
        if (q.count("yaw") && parse_num(q["yaw"], v)) { yaw_ = std::remainder(v, 360.0); camera = true; }
        if (q.count("pitch") && parse_num(q["pitch"], v)) { pitch_ = std::remainder(v, 360.0); camera = true; }
        if (q.count("zoom") && parse_num(q["zoom"], v) && v > 0) { zoom_ = std::clamp(v, 1e-4, 1e6); camera = true; }
        if (q.count("fit")) { base_extent_ = 0; zoom_ = 1; ++fit_gen_; camera = true; }
        if (q.count("color")) {
            color_ = q["color"] == "speed" ? ColorMode::speed : ColorMode::density;
            camera = true;
        }
        if (q.count("size")) {
            const std::string& s = q["size"];
            size_t x = s.find('x');
            double w = 0, h = 0;
            if (x != std::string::npos && parse_num(s.substr(0, x), w) && parse_num(s.substr(x + 1), h)) {
                // Keep the scale (world units per pixel) when the window changes width.
                if (base_extent_ > 0 && width_ > 0) base_extent_ *= std::clamp(w, 64.0, 4096.0) / width_;
                width_ = static_cast<int>(std::clamp(w, 64.0, 4096.0));
                height_ = static_cast<int>(std::clamp(h, 64.0, 4096.0));
                camera = true;
            }
        }
        if (q.count("pause")) { paused_ = q["pause"] != "0"; run = true; }
        if (q.count("step")) { step_once_ = true; paused_ = true; run = true; }
        if (q.count("stop")) { stop_ = true; run = true; }
        if (camera) camera_dirty_ = true;
    }
    if (camera) render_cv_.notify_one();
    if (run) control_cv_.notify_all();
    return status_json();
}

namespace {

const char* const kPage = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>lettuce live</title>
<style>
:root{--bg:#07080d;--panel:#11131c;--line:#252a3a;--text:#d8dcea;--dim:#8a91a8;--accent:#ff9a3c;--bad:#ff5a5a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
body{display:flex;flex-direction:column}
header{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line)}
header b{color:var(--accent);font-weight:600}.k{color:var(--dim)}
#bar{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
button{background:#1b1f2d;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}button.on{background:var(--accent);color:#111;border-color:var(--accent)}
#stop{color:var(--bad)}
main{flex:1;position:relative;overflow:hidden;cursor:grab;touch-action:none}main.drag{cursor:grabbing}
#view{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;user-select:none;-webkit-user-drag:none}
#msg{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);color:var(--dim)}
footer{padding:4px 12px;color:var(--dim);background:var(--panel);border-top:1px solid var(--line)}
</style></head><body>
<header>
 <span><b>lettuce</b> <span id="scn"></span></span>
 <span><span class="k">n</span> <span id="n">-</span></span>
 <span><span class="k">step</span> <span id="step">-</span></span>
 <span><span class="k">t</span> <span id="t">-</span></span>
 <span><span class="k">dE/E0</span> <span id="de">-</span></span>
 <span><span class="k">2K/|W|</span> <span id="vir">-</span></span>
 <span><span class="k">ms/step</span> <span id="ms">-</span></span>
 <span id="state"></span>
 <span id="bar">
  <button id="pause" title="space">Pause</button><button id="step1" title="s">Step</button>
  <button id="fit" title="f">Fit</button><button id="color" title="c">Colour: density</button>
  <button id="stop">Stop</button>
 </span>
</header>
<main id="main"><div id="msg">waiting for the first frame...</div><img id="view" alt=""></main>
<footer>drag to rotate · scroll to zoom · <span id="info"></span></footer>
<script>
const $=id=>document.getElementById(id);
const img=$('view'),main=$('main');
let st={yaw:0,pitch:0,zoom:1,paused:false,color:'density'},online=true;
function fmt(x,p=4){return (x===null||x===undefined||!isFinite(x))?'-':(Math.abs(x)>=1e4||(Math.abs(x)<1e-3&&x!==0))?x.toExponential(2):(+x).toPrecision(p)}
function show(s){
 st.yaw=s.yaw;st.pitch=s.pitch;st.zoom=s.zoom;st.paused=s.paused;st.color=s.color;
 $('scn').textContent=s.scenario+' · '+s.method;
 $('n').textContent=s.n.toLocaleString();
 $('step').textContent=s.step.toLocaleString()+(s.last_step?' / '+s.last_step.toLocaleString():'');
 $('t').textContent=fmt(s.time);$('de').textContent=s.diag_step||s.step?fmt(s.de,3):'-';
 $('vir').textContent=fmt(s.virial,4);$('ms').textContent=fmt(s.ms_per_step,3);
 $('state').textContent=s.stopped?'stopped':s.finished?'finished':s.paused?'paused':'running';
 $('pause').textContent=s.paused?'Run':'Pause';$('pause').classList.toggle('on',s.paused);
 $('color').textContent='Colour: '+s.color;
 $('info').textContent=s.backend+' · '+s.width+'x'+s.height+' · '+fmt(s.fps,3)+' fps · render '+fmt(s.render_ms,3)+' ms';
}
async function ctl(q){try{const r=await fetch('/control?'+q);show(await r.json())}catch(e){}}
async function poll(){try{const r=await fetch('/status');show(await r.json());if(!online){online=true;connect()}}
 catch(e){online=false;$('state').textContent='disconnected'}}
setInterval(poll,250);poll();
function connect(){img.src='/stream?'+Date.now()}
img.onload=()=>{$('msg').style.display='none'};
img.onerror=()=>{$('msg').style.display='';$('msg').textContent='stream lost, retrying...';setTimeout(connect,1000)};
connect();
let pend=null,timer=0;
function send(q){pend=q;if(!timer)timer=setTimeout(()=>{timer=0;const p=pend;pend=null;ctl(p)},40)}
let drag=null;
main.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};main.setPointerCapture(e.pointerId);main.classList.add('drag')});
main.addEventListener('pointerup',e=>{drag=null;main.classList.remove('drag')});
main.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};
 st.yaw-=dx*0.4;st.pitch=Math.max(-180,Math.min(180,st.pitch-dy*0.4));send('yaw='+st.yaw.toFixed(2)+'&pitch='+st.pitch.toFixed(2))});
main.addEventListener('wheel',e=>{e.preventDefault();st.zoom*=Math.exp(-e.deltaY*0.0015);send('zoom='+st.zoom.toPrecision(6))},{passive:false});
$('pause').onclick=()=>ctl('pause='+(st.paused?0:1));
$('step1').onclick=()=>ctl('step=1');
$('fit').onclick=()=>ctl('fit=1');
$('color').onclick=()=>ctl('color='+(st.color==='density'?'speed':'density'));
$('stop').onclick=()=>{if(confirm('Stop the simulation? It ends the run (outputs so far are kept).'))ctl('stop=1')};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 if(e.key===' '){e.preventDefault();$('pause').click()}else if(e.key==='s')$('step1').click();
 else if(e.key==='f')$('fit').click();else if(e.key==='c')$('color').click()});
let rs=0;new ResizeObserver(()=>{clearTimeout(rs);rs=setTimeout(()=>{const d=Math.min(window.devicePixelRatio||1,2);
 const w=Math.round(Math.min(1920,main.clientWidth*d)),h=Math.round(Math.min(1920,main.clientHeight*d));if(w>63&&h>63)ctl('size='+w+'x'+h)},250)}).observe(main);
</script></body></html>
)HTML";

}  // namespace
