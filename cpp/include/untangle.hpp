#pragma once
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/GraphAttributes.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tmap {

struct UntangleParams {
    bool rotate = true;          // false => reflect-only
    int max_sub = 2000;          // cap on subtree size considered (0 => no cap = n)
    int passes = 4;              // greedy sweeps
    int rot_steps = 64;          // rotation angles per subtree
    double max_angle = 1.57079632679489662;  // radians (pi/2); <=0 or >=pi => no cap
    double slide_eps = 0.0;      // bounded stem-slide fraction (0 => off, exact lengths)
    int slide_steps = 5;         // stem-scale samples in [1-eps, 1+eps] when eps>0
};

namespace untangle_detail {

inline int isgn(double v) { return (v > 0.0) - (v < 0.0); }

inline bool segments_cross(double ax, double ay, double bx, double by, double cx,
        double cy, double dx, double dy) {
    const int o1 = isgn((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
    const int o2 = isgn((bx - ax) * (dy - ay) - (by - ay) * (dx - ax));
    const int o3 = isgn((dx - cx) * (ay - cy) - (dy - cy) * (ax - cx));
    const int o4 = isgn((dx - cx) * (by - cy) - (dy - cy) * (bx - cx));
    return o1 != o2 && o3 != o4;
}

struct SegGrid {
    double cell = 1.0, x0 = 0.0, y0 = 0.0;
    std::unordered_map<long long, std::vector<int>> cells;

    std::vector<std::vector<long long>> ekeys;

    static long long pack(int ix, int iy) {
        return (static_cast<long long>(ix + (1 << 30)) << 31) |
                static_cast<long long>(iy + (1 << 30));
    }
    int cix(double v) const { return static_cast<int>(std::floor((v - x0) / cell)); }
    int ciy(double v) const { return static_cast<int>(std::floor((v - y0) / cell)); }

    template <class F>
    void forKeys(double ax, double ay, double bx, double by, F&& f) const {
        const int ix0 = cix(std::min(ax, bx)), ix1 = cix(std::max(ax, bx));
        const int iy0 = ciy(std::min(ay, by)), iy1 = ciy(std::max(ay, by));
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                f(pack(ix, iy));
            }
        }
    }
    void insert(int i, double ax, double ay, double bx, double by) {
        forKeys(ax, ay, bx, by, [&](long long k) {
            cells[k].push_back(i);
            ekeys[i].push_back(k);
        });
    }
    void remove(int i) {
        for (long long k : ekeys[i]) {
            auto& v = cells[k];
            for (size_t t = 0; t < v.size(); ++t) {
                if (v[t] == i) {
                    v[t] = v.back();
                    v.pop_back();
                    break;
                }
            }
        }
        ekeys[i].clear();
    }
};

}

inline void untangle_post_pass(const ogdf::Graph& g, ogdf::GraphAttributes& ga,
        const UntangleParams& params, std::mt19937& /*rng*/) {
    using namespace untangle_detail;
    const int n = g.numberOfNodes();
    const int m = g.numberOfEdges();
    if (n < 3 || m < 2) {
        return;
    }
    constexpr double kTwoPi = 6.28318530717958647692;
    constexpr double kPi = 3.14159265358979323846;

    std::vector<double> x(n), y(n);
    for (ogdf::node v : g.nodes) {
        x[v->index()] = ga.x(v);
        y[v->index()] = ga.y(v);
    }
    std::vector<int> eu(m), ev(m);
    std::vector<std::vector<std::pair<int, int>>> adj(n);
    {
        int i = 0;
        for (ogdf::edge e : g.edges) {
            const int a = e->source()->index(), b = e->target()->index();
            eu[i] = a;
            ev[i] = b;
            adj[a].push_back({b, i});
            adj[b].push_back({a, i});
            ++i;
        }
    }

    std::vector<int> parent(n, -1), parentEdge(n, -1), order, tin(n, 0), tout(n, 0);
    order.reserve(n);
    std::vector<char> vis(n, 0);
    int clock = 0;
    auto dfs = [&](int start) {
        std::vector<std::pair<int, bool>> st;
        vis[start] = 1;
        st.push_back({start, false});
        while (!st.empty()) {
            const auto [u, closing] = st.back();
            st.pop_back();
            if (closing) {
                tout[u] = clock;
                continue;
            }
            tin[u] = clock;
            order.push_back(u);
            ++clock;
            st.push_back({u, true});
            for (const auto& [w, ei] : adj[u]) {
                if (!vis[w]) {
                    vis[w] = 1;
                    parent[w] = u;
                    parentEdge[w] = ei;
                    st.push_back({w, false});
                }
            }
        }
    };
    int root0 = 0;
    for (int v = 1; v < n; ++v) {
        if (adj[v].size() > adj[root0].size()) {
            root0 = v;
        }
    }
    dfs(root0);
    for (int s = 0; s < n; ++s) {
        if (!vis[s]) {
            dfs(s);
        }
    }

    std::vector<double> len;
    len.reserve(m);
    for (int i = 0; i < m; ++i) {
        const double L = std::hypot(x[eu[i]] - x[ev[i]], y[eu[i]] - y[ev[i]]);
        if (L > 0.0) {
            len.push_back(L);
        }
    }
    double med = 1.0;
    if (!len.empty()) {
        std::nth_element(len.begin(), len.begin() + len.size() / 2, len.end());
        med = len[len.size() / 2];
    }
    double minX = x[0], minY = y[0];
    for (int v = 1; v < n; ++v) {
        minX = std::min(minX, x[v]);
        minY = std::min(minY, y[v]);
    }
    SegGrid grid;
    grid.cell = std::max(med * 2.0, 1e-9);
    grid.x0 = minX;
    grid.y0 = minY;
    grid.ekeys.assign(m, {});
    for (int i = 0; i < m; ++i) {
        grid.insert(i, x[eu[i]], y[eu[i]], x[ev[i]], y[ev[i]]);
    }

    const int cap = (params.max_sub > 0) ? params.max_sub : n;
    std::vector<int> cand;
    for (int v = 0; v < n; ++v) {
        const int sz = tout[v] - tin[v];
        if (parent[v] != -1 && sz >= 2 && sz <= cap) {
            cand.push_back(v);
        }
    }
    std::sort(cand.begin(), cand.end(),
            [&](int a, int b) { return (tout[a] - tin[a]) < (tout[b] - tin[b]); });

    const bool rotate = params.rotate;
    const int K = std::max(1, params.rot_steps);
    const bool capped = (params.max_angle > 0.0 && params.max_angle < kPi);
    const double angleCap = capped ? params.max_angle : kPi;

    std::vector<double> slideFactors;
    if (rotate && params.slide_eps > 0.0) {
        const int ss = std::max(2, params.slide_steps);
        for (int i = 0; i < ss; ++i) {
            slideFactors.push_back(1.0 + params.slide_eps * (2.0 * i / (ss - 1) - 1.0));
        }
    } else {
        slideFactors.push_back(1.0);
    }

    std::vector<char> inMoved(m, 0);
    std::vector<long long> stamp(m, -1);
    long long token = 0;
    std::vector<double> tx(n), ty(n), bx(n), by(n);

    auto probe = [&](const std::vector<int>& pe, const std::vector<double>& qx,
                         const std::vector<double>& qy) -> long long {
        long long cnt = 0;
        for (int i : pe) {
            ++token;
            const double ax = qx[eu[i]], ay = qy[eu[i]], bxp = qx[ev[i]], byp = qy[ev[i]];
            const int ni0 = eu[i], ni1 = ev[i];
            grid.forKeys(ax, ay, bxp, byp, [&](long long k) {
                auto it = grid.cells.find(k);
                if (it == grid.cells.end()) {
                    return;
                }
                for (int j : it->second) {
                    if (inMoved[j] || stamp[j] == token) {
                        continue;
                    }
                    stamp[j] = token;
                    const int nj0 = eu[j], nj1 = ev[j];
                    if (ni0 == nj0 || ni0 == nj1 || ni1 == nj0 || ni1 == nj1) {
                        continue;
                    }
                    if (segments_cross(ax, ay, bxp, byp, x[nj0], y[nj0], x[nj1], y[nj1])) {
                        ++cnt;
                    }
                }
            });
        }
        return cnt;
    };

    for (int pass = 0; pass < std::max(1, params.passes); ++pass) {
        long long gain = 0;
        for (int cc : cand) {
            const int a0 = tin[cc], a1 = tout[cc];
            const int par = parent[cc];
            std::vector<int> pe;
            pe.reserve(a1 - a0 + 1);
            for (int t = a0; t < a1; ++t) {
                const int v = order[t];
                if (v != cc) {
                    pe.push_back(parentEdge[v]);
                }
            }
            pe.push_back(parentEdge[cc]);
            for (int i : pe) {
                inMoved[i] = 1;
            }

            const double px = x[par], py = y[par];
            const long long xOld = probe(pe, x, y);
            if (xOld == 0) {
                for (int i : pe) {
                    inMoved[i] = 0;
                }
                continue;
            }

            long long best = xOld;
            double bestDisp = 0.0;
            bool have = false;

            auto evalAndKeep = [&](auto&& xform) {
                double disp = 0.0;
                for (int t = a0; t < a1; ++t) {
                    const int v = order[t];
                    double ox, oy;
                    xform(x[v] - px, y[v] - py, ox, oy);
                    tx[v] = px + ox;
                    ty[v] = py + oy;
                    const double mdx = tx[v] - x[v], mdy = ty[v] - y[v];
                    disp += mdx * mdx + mdy * mdy;
                }
                tx[par] = px;
                ty[par] = py;
                const long long xr = probe(pe, tx, ty);
                if (xr < best || (xr == best && have && disp < bestDisp)) {
                    best = xr;
                    bestDisp = disp;
                    have = true;
                    for (int t = a0; t < a1; ++t) {
                        const int v = order[t];
                        bx[v] = tx[v];
                        by[v] = ty[v];
                    }
                }
            };

            double dcx = x[cc] - px, dcy = y[cc] - py;
            const double nrm = std::hypot(dcx, dcy);
            if (nrm > 1e-12 && (!capped || !rotate)) {
                dcx /= nrm;
                dcy /= nrm;
                evalAndKeep([&](double vx, double vy, double& ox, double& oy) {
                    const double d = vx * dcx + vy * dcy;
                    ox = 2.0 * d * dcx - vx;
                    oy = 2.0 * d * dcy - vy;
                });
            }

            if (rotate) {
                const double sx0 = x[cc] - px, sy0 = y[cc] - py;
                for (int k = 0; k < K; ++k) {
                    const double th = kTwoPi * k / K;
                    const double turn = (th > kPi) ? th - kTwoPi : th;
                    if (k > 0 && std::abs(turn) > angleCap) {
                        continue;
                    }
                    const double ct = std::cos(th), sstn = std::sin(th);
                    const double rcx = ct * sx0 - sstn * sy0;
                    const double rcy = sstn * sx0 + ct * sy0;
                    for (double s : slideFactors) {
                        if (k == 0 && s == 1.0) {
                            continue;
                        }
                        const double ttx = (s - 1.0) * rcx, tty = (s - 1.0) * rcy;
                        evalAndKeep([&](double vx, double vy, double& ox, double& oy) {
                            ox = ct * vx - sstn * vy + ttx;
                            oy = sstn * vx + ct * vy + tty;
                        });
                    }
                }
            }

            if (have) {
                for (int i : pe) {
                    grid.remove(i);
                }
                for (int t = a0; t < a1; ++t) {
                    const int v = order[t];
                    x[v] = bx[v];
                    y[v] = by[v];
                }
                for (int i : pe) {
                    grid.insert(i, x[eu[i]], y[eu[i]], x[ev[i]], y[ev[i]]);
                }
                gain += (xOld - best);
            }
            for (int i : pe) {
                inMoved[i] = 0;
            }
        }
        if (gain == 0) {
            break;
        }
    }

    for (ogdf::node v : g.nodes) {
        ga.x(v) = x[v->index()];
        ga.y(v) = y[v->index()];
    }
}

}  // namespace tmap
