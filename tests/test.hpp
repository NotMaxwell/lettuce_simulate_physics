// Minimal self-registering test harness: TEST(name) { CHECK(cond); CHECK_NEAR(a, b, tol); }
// A test can call SKIP("reason") (e.g. no CUDA device) to count as skipped, not passed.
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace t {
struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
inline int& failures() { static int f = 0; return f; }
struct Skip { std::string why; };
}  // namespace t

#define T_CAT2(a, b) a##b
#define T_CAT(a, b) T_CAT2(a, b)
#define TEST(name)                                                   \
    static void T_CAT(test_fn_, __LINE__)();                         \
    static t::Reg T_CAT(test_reg_, __LINE__)(name, T_CAT(test_fn_, __LINE__)); \
    static void T_CAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            ++t::failures();                                                              \
        }                                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                  \
    do {                                                                                       \
        double t_a = (a), t_b = (b), t_tol = (tol);                                            \
        if (!(std::fabs(t_a - t_b) <= t_tol)) {                                                \
            std::fprintf(stderr, "  %s:%d: CHECK_NEAR(%s, %s, %s) failed: %.9g vs %.9g (diff %.3g)\n", \
                         __FILE__, __LINE__, #a, #b, #tol, t_a, t_b, std::fabs(t_a - t_b));   \
            ++t::failures();                                                                   \
        }                                                                                      \
    } while (0)

#define SKIP(why) throw t::Skip{why}
