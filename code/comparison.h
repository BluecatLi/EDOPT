#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "projection.h"

// from the interaction matrix and current depth calculate the desired state
// change to move the image by approximately 1 pixel.

//[du dv] =
//[ fx/d, 0, -(u-cx)/d, -(u-cx)(v-cy)/fy, fx*fx+(u-cx)(u-cx)/fx, -(v-cy)fx/fy
//  0, fy/d, -(v-cy)/d, -(fy*fy)-(v-cy)(v-cy)/fy, (u-cx)(v-cy)/fx, (u-cx)fy/fx] *

// [dx, dy, dz, dalpha, dbeta, dgamma]

class warpManager {

public:

    //internal definitions
    enum cam_param_name{w,h,cx,cy,fx,fy};
    enum warp_name{xp, yp, zp, ap, bp, cp, xn, yn, zn, an, bn, cn,
                   xp2, yp2, zp2, ap2, bp2, cp2, xn2, yn2, zn2, an2, bn2, cn2};
    enum axis_name{x=0,y=1,z=2,a=3,b=4,c=5};
    typedef struct warp_bundle
    {
        cv::Mat M;
        cv::Mat rmp;
        cv::Mat rmsp;
        cv::Mat img_warp;
        int axis{0};
        double delta{0.0};
        double score{-DBL_MAX};
        bool active{false};
    } warp_bundle;

    //fixed parameters to set
    std::array<double, 6> cam;
    cv::Size proc_size{cv::Size(100, 100)};

    //parameters that must be udpated externally
    double scale{1.0};
    cv::Mat proc_obs;

    //internal variables
    warp_bundle projection;
    std::array<warp_bundle, 24> warps;
    std::deque<const warp_bundle *> warp_history;
    cv::Mat prmx;
    cv::Mat prmy;
    cv::Mat nrmx;
    cv::Mat nrmy;

    //output
    std::array<double, 7> state_current;

public:

    void initialise(int size_to_process, bool dp2)
    {
        proc_size = cv::Size(size_to_process, size_to_process);

        projection.axis = -1;
        projection.delta = 0;
        proc_obs = cv::Mat::zeros(proc_size, CV_32F);
        projection.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        for(auto &warp : warps) {
            warp.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        }

        prmx = cv::Mat::zeros(proc_size, CV_32F);
        prmy = cv::Mat::zeros(proc_size, CV_32F);
        nrmx = cv::Mat::zeros(proc_size, CV_32F);
        nrmy = cv::Mat::zeros(proc_size, CV_32F);

        warps[xp].active = warps[xn].active = true; 
        warps[yp].active = warps[yn].active = true; 
        warps[zp].active = warps[zn].active = true; 
        warps[ap].active = warps[an].active = true; 
        warps[bp].active = warps[bn].active = true; 
        warps[cp].active = warps[cn].active = true;

        if (dp2) {
            warps[xp2].active = warps[xn2].active = true;
            warps[yp2].active = warps[yn2].active = true;
            warps[zp2].active = warps[zn2].active = true;
            warps[ap2].active = warps[an2].active = true;
            warps[bp2].active = warps[bn2].active = true;
            warps[cp2].active = warps[cn2].active = true;
        }
    }

    void create_m_x(double dp, warp_name p, warp_name n)
    {
        //x shift by dp
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                //positive
                prmx.at<float>(y, x) = x-dp;
                prmy.at<float>(y, x) = y;
                //negative
                nrmx.at<float>(y, x) = x+dp;
                nrmy.at<float>(y, x) = y;
            }
        }
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = x; warps[n].axis = x;
        warps[p].delta = dp; warps[n].delta = -dp;
    }

    void create_m_y(double dp, warp_name p, warp_name n)
    {
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                //positive
                prmx.at<float>(y, x) = x;
                prmy.at<float>(y, x) = y-dp;
                //negative
                nrmx.at<float>(y, x) = x;
                nrmy.at<float>(y, x) = y+dp;
            }
        }
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = y; warps[n].axis = y;
        warps[p].delta = dp; warps[n].delta = -dp;
    }

    void create_m_z(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dx = -(x-cx) * dp / proc_size.width;
                double dy = -(y-cy) * dp / proc_size.height;
                //positive
                prmx.at<float>(y, x) = x - dx;
                prmy.at<float>(y, x) = y - dy;
                //negative
                nrmx.at<float>(y, x) = x + dx;
                nrmy.at<float>(y, x) = y + dy;
            }
        }
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = z; warps[n].axis = z;
        warps[p].delta = dp / proc_size.width; 
        warps[n].delta = -dp / proc_size.width;
    }

    void create_m_a(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
        double theta = M_PI_2 * dp / (proc_size.height * 0.5);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dy = -dp * cos(0.5 * M_PI * (y - cy) / (proc_size.height * 0.5));
                //positive
                prmx.at<float>(y, x) = x;
                prmy.at<float>(y, x) = y - dy;
                //negative
                nrmx.at<float>(y, x) = x;
                nrmy.at<float>(y, x) = y + dy;
            }
        }
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = a; warps[n].axis = a;
        warps[p].delta = theta; warps[n].delta = -theta;
    }

    void create_m_b(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
        double theta = M_PI_2 * dp / (proc_size.width * 0.5);
        
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dx = -dp * cos(0.5 * M_PI * (x - cx) / (proc_size.width * 0.5));
                //positive
                prmx.at<float>(y, x) = x - dx;
                prmy.at<float>(y, x) = y;
                //negative
                nrmx.at<float>(y, x) = x + dx;
                nrmy.at<float>(y, x) = y;
            }
        }
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = b; warps[n].axis = b;
        warps[p].delta = theta; warps[n].delta = -theta;
    }

    void create_m_c(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
        double theta = atan2(dp, std::max(proc_size.width, proc_size.height)*0.5);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dx = -(y - cy) * cam[fx] / cam[fy] * theta;
                double dy =  (x - cx) * cam[fy] / cam[fx] * theta;
                //positive
                prmx.at<float>(y, x) = x - dx;
                prmy.at<float>(y, x) = y - dy;
                //negative
                nrmx.at<float>(y, x) = x + dx;
                nrmy.at<float>(y, x) = y + dy;
            }
        }
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = c; warps[n].axis = c;
        warps[p].delta = theta; warps[n].delta = -theta;
    }

    void create_Ms(double dp)
    {
        create_m_x(dp, xp, xn);
        create_m_y(dp, yp, yn);
        create_m_z(dp, zp, zn);
        create_m_a(dp, ap, an);
        create_m_b(dp, bp, bn);
        create_m_c(dp, cp, cn);

        create_m_x(dp*2, xp2, xn2);
        create_m_y(dp*2, yp2, yn2);
        create_m_z(dp*2, zp2, zn2);
        create_m_a(dp*2, ap2, an2);
        create_m_b(dp*2, bp2, bn2);
        create_m_c(dp*2, cp2, cn2);
    }

    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
    }

    void make_predictive_warps()
    {
        for(auto &warp : warps) {
            if(warp.axis == a || warp.axis == b) continue;
            if(warp.active)
                cv::remap(projection.img_warp, warp.img_warp, warp.rmp, warp.rmsp, cv::INTER_LINEAR);
        }
    }

    void warp_by_history(cv::Mat &image)
    {
        for(auto warp : warp_history)
            cv::remap(image, image, warp->rmp, warp->rmsp, cv::INTER_LINEAR);
        warp_history.clear();
    }

double similarity_score(const cv::Mat &O, const cv::Mat &E) {
    return cv::sum(E.mul(O))[0];
}

    void score_predictive_warps()
    {
        projection.score = similarity_score(proc_obs, projection.img_warp);
        projection.score = projection.score < 0 ? 0 : projection.score;
        for(auto &w : warps)
            if(w.active) w.score = similarity_score(proc_obs, w.img_warp);
    }

    // apply a continuous delta (same units as warp.delta) on one axis,
    // using the exact same per-axis conversion as the discrete update.
    void apply_axis_delta(int axis, double delta)
    {
        double d = fabs(state_current[z]);
        switch(axis) {
            case(x): state_current[x] += delta * scale * d / cam[fx]; break;
            case(y): state_current[y] += delta * scale * d / cam[fy]; break;
            case(z): state_current[z] += delta * d;                   break;
            case(a): perform_rotation(state_current, 0, delta);       break;
            case(b): perform_rotation(state_current, 1, delta);       break;
            case(c): perform_rotation(state_current, 2, delta);       break;
        }
    }

    void update_state(const warp_bundle &best)
    {
        apply_axis_delta(best.axis, best.delta);
        //cv::remap(projection.img_warp, projection.img_warp, best.rmp, best.rmsp, cv::INTER_LINEAR);
        //warp_history.push_back(&best);
    }

    // Continuous variable-magnitude step per axis, from the already-computed
    // scores s(0)=projection.score, s(+/-delta)=warps[p/n].score.
    //  - concave (peak bracketed): damped-Newton parabola-vertex step
    //  - non-concave (still climbing => object moved far): saturated trust step
    // This removes the fixed +/-1px cap that causes the fast-motion staircase.
    bool update_parabolic(double lambda = 1e-3, double trust = 3.0, double eps = 0.02)
    {
        struct Ax { int axis; int p; int n; };
        static const Ax axes[6] = {
            {x, xp, xn}, {y, yp, yn}, {z, zp, zn},
            {a, ap, an}, {b, bp, bn}, {c, cp, cn}
        };
        bool updated = false;
        double s0 = projection.score;
        for (const auto &ax : axes) {
            if (!warps[ax.p].active || !warps[ax.n].active) continue;
            double sp = warps[ax.p].score;
            double sn = warps[ax.n].score;
            double D  = warps[ax.p].delta;            // +base step (axis units)
            if (D == 0.0) continue;

            double concav = 2.0*s0 - sp - sn;         // >0 => concave peak
            double grad   = sp - sn;                  // ascent direction sign

            // deadband: ignore axes whose +/- scores differ only by noise.
            // scale threshold by the local score level so it adapts to event rate.
            double ref = (s0 > 0.0 ? s0 : 0.5*(sp+sn));
            double grad_thresh = eps * (ref > 0.0 ? ref : 1.0);
            if (fabs(grad) < grad_thresh) continue;   // no reliable signal -> hold

            double step;
            if (concav > 1e-9) {
                step = D * grad / (2.0*concav + lambda);   // Newton vertex
            } else {
                step = (grad > 0 ? 1.0 : -1.0) * trust * D; // saturate (signal already passed deadband)
            }
            double R = trust * fabs(D);                // trust-region clamp
            if (step >  R) step =  R;
            if (step < -R) step = -R;
            if (fabs(step) < 1e-6 * fabs(D)) continue; // negligible

            apply_axis_delta(ax.axis, step);
            updated = true;
        }
        return updated;
    }
    // ---- 1D parabolic step (exact same logic as update_parabolic, factored out) ----
    bool parabolic_step_1d(double s0, double sp, double sn, double D,
                           double lambda, double trust, double eps, double &step)
    {
        if (D == 0.0) return false;
        double concav = 2.0*s0 - sp - sn;
        double grad   = sp - sn;
        double ref = (s0 > 0.0 ? s0 : 0.5*(sp+sn));
        double grad_thresh = eps * (ref > 0.0 ? ref : 1.0);
        if (fabs(grad) < grad_thresh) return false;
        if (concav > 1e-9) step = D * grad / (2.0*concav + lambda);
        else               step = (grad > 0 ? 1.0 : -1.0) * trust * D;
        double R = trust * fabs(D);
        if (step >  R) step =  R;
        if (step < -R) step = -R;
        if (fabs(step) < 1e-6 * fabs(D)) return false;
        return true;
    }

    // ---- corner score s(+d_affine, +d_rot): apply the affine axis' +delta remap
    //      to the already-rendered +delta rotation template, then score. ----
    //      NOTE: sequential_loop only (needs rendered a/b templates in warps[ap/bp]).
    double score_corner(int affine_pos, int rendered_pos)
    {
        static cv::Mat corner;
        cv::remap(warps[rendered_pos].img_warp, corner,
                  warps[affine_pos].rmp, warps[affine_pos].rmsp, cv::INTER_LINEAR);
        return similarity_score(proc_obs, corner);
    }

    // ---- coupled 2x2 quadratic on an (affine, rotation) pair ----
    //      af in {x,y}, ro in {b(yaw), a(pitch)}. Works in t-units (multiples of
    //      each axis' own base step), so the 1D reduction matches update_parabolic.
bool solve_pair(int af, int ro, double lambda, double trust, double eps,
                    double *out_affine = nullptr)
    {
        if (out_affine) *out_affine = 0.0;
        const int afp = af, afn = af + 6;
        const int rop = ro, ron = ro + 6;
        if (!warps[afp].active || !warps[rop].active) return false;

        const double s0   = projection.score;
        const double sa_p = warps[afp].score, sa_n = warps[afn].score;
        const double sb_p = warps[rop].score, sb_n = warps[ron].score;
        const double Da = warps[afp].delta, Db = warps[rop].delta;
        if (Da == 0.0 || Db == 0.0) return false;

        const double ref = (s0 > 0.0 ? s0 : 0.25*(sa_p+sa_n+sb_p+sb_n));
        const double thr = eps * (ref > 0.0 ? ref : 1.0);
        const bool sig_a = fabs(sa_p - sa_n) >= thr;
        const bool sig_b = fabs(sb_p - sb_n) >= thr;
        if (!sig_a && !sig_b) return false;

        if (sig_a != sig_b) {                       // slow motion -> 1D, no leak
            double step;
            if (sig_a && parabolic_step_1d(s0, sa_p, sa_n, Da, lambda, trust, eps, step)) {
                apply_axis_delta(af, step); if (out_affine) *out_affine = step; return true;
            }
            if (sig_b && parabolic_step_1d(s0, sb_p, sb_n, Db, lambda, trust, eps, step)) {
                apply_axis_delta(ro, step); return true;      // rotation only
            }
            return false;
        }

        const double ga = 0.5*(sa_p - sa_n), gb = 0.5*(sb_p - sb_n);
        const double A = (2.0*s0 - sa_p - sa_n) + lambda;
        const double B = (2.0*s0 - sb_p - sb_n) + lambda;
        double ta, tb;
        if (A > 1e-9 && B > 1e-9) {
            double Cab = sa_p + sb_p - score_corner(afp, rop) - s0;
            double det = A*B - Cab*Cab;
            if (det < 0.1*A*B) { Cab = 0.0; det = A*B; }     // conditioning guard
            ta = ( B*ga - Cab*gb) / det;
            tb = (-Cab*ga +  A*gb) / det;
        } else { ta = (ga>0?trust:-trust); tb = (gb>0?trust:-trust); }
        if (ta> trust) ta= trust; if (ta<-trust) ta=-trust;
        if (tb> trust) tb= trust; if (tb<-trust) tb=-trust;

        bool moved = false;
        if (fabs(ta) > 1e-6) { apply_axis_delta(af, ta*Da); if(out_affine)*out_affine=ta*Da; moved=true; }
        if (fabs(tb) > 1e-6) { apply_axis_delta(ro, tb*Db); moved=true; }
        return moved;
    }
    // ---- coupled continuous update: pairs {x,yaw} {y,pitch}, z & roll stay 1D ----
    bool update_parabolic_coupled(double lambda = 1e-3, double trust = 3.0, double eps = 0.02)
    {
        bool updated = false;
        updated |= solve_pair(x, b, lambda, trust, eps);   // x  <-> yaw
        updated |= solve_pair(y, a, lambda, trust, eps);   // y  <-> pitch

        double step;
        if (warps[zp].active && warps[zn].active &&
            parabolic_step_1d(projection.score, warps[zp].score, warps[zn].score,
                              warps[zp].delta, lambda, trust, eps, step)) {
            apply_axis_delta(z, step); updated = true;
        }
        if (warps[cp].active && warps[cn].active &&
            parabolic_step_1d(projection.score, warps[cp].score, warps[cn].score,
                              warps[cp].delta, lambda, trust, eps, step)) {
            apply_axis_delta(c, step); updated = true;
        }
        return updated;
    }
// re-center projection.img_warp by a continuous step on ONE affine axis,
    // using the same field geometry as create_m_* (assumes square proc_size).
    // step is in warp.delta units (px for x/y; dp/W for z; rad for c).
    void warp_projection_by_step(int axis, double step)
    {
        if (step == 0.0) return;
        const double cx = proc_size.width * 0.5, cy = proc_size.height * 0.5;
        for (int yy = 0; yy < proc_size.height; ++yy)
            for (int xx = 0; xx < proc_size.width; ++xx) {
                double mx = xx, my = yy;
                switch (axis) {
                    case x: mx = xx - step;                       break;
                    case y: my = yy - step;                       break;
                    case z: mx = xx + (xx-cx)*step;
                            my = yy + (yy-cy)*step;               break;
                    case c: mx = xx + (yy-cy)*cam[fx]/cam[fy]*step;
                            my = yy - (xx-cx)*cam[fy]/cam[fx]*step; break;
                    default: return;                              // a/b: render-only
                }
                prmx.at<float>(yy, xx) = (float)mx;
                prmy.at<float>(yy, xx) = (float)my;
            }
        static cv::Mat tmp;                                       // avoid in-place remap
        cv::remap(projection.img_warp, tmp, prmx, prmy, cv::INTER_LINEAR);
        tmp.copyTo(projection.img_warp);
    }

    // re-score projection + affine +/-delta templates only (a/b stay frozen).
    void score_affine()
    {
        projection.score = similarity_score(proc_obs, projection.img_warp);
        if (projection.score < 0) projection.score = 0;
        const int aff[8] = {xp,xn,yp,yn,zp,zn,cp,cn};
        for (int i : aff)
            if (warps[i].active) warps[i].score = similarity_score(proc_obs, warps[i].img_warp);
    }

    // 1D parabolic step on one axis; reports the applied step (warp.delta units).
    bool solve_axis_track(int axis, double lambda, double trust, double eps, double &out_step)
    {
        out_step = 0.0;
        const int p = axis, n = axis + 6;
        if (!warps[p].active || !warps[n].active) return false;
        double step;
        if (!parabolic_step_1d(projection.score, warps[p].score, warps[n].score,
                               warps[p].delta, lambda, trust, eps, step)) return false;
        apply_axis_delta(axis, step);
        out_step = step;
        return true;
    }

    // Solve the frame to convergence: 1 coupled sweep (fresh render) + affine
    // inner iterations on a re-warped projection (pitch/yaw frozen).
    // --inner 1  ==  --cpair.
    bool update_iterative(double lambda = 1e-3, double trust = 3.0,
                          double eps = 0.02, int max_inner = 3)
    {
        bool any = false; double s;

        // iter 0: full coupled sweep (scores already computed by caller).
        if (solve_pair(x, b, lambda, trust, eps, &s)) { warp_projection_by_step(x, s); any = true; }
        if (solve_pair(y, a, lambda, trust, eps, &s)) { warp_projection_by_step(y, s); any = true; }
        if (solve_axis_track(z, lambda, trust, eps, s)) { warp_projection_by_step(z, s); any = true; }
        if (solve_axis_track(c, lambda, trust, eps, s)) { warp_projection_by_step(c, s); any = true; }

        // inner iters: affine block only, Jacobi update on re-centered templates.
        for (int it = 1; it < max_inner; ++it) {
            make_predictive_warps();        // regen x/y/z/c +/- (skips a/b)
            score_affine();
            double dx, dy, dz, dc; bool moved = false;
            bool mX = solve_axis_track(x, lambda, trust, eps, dx);
            bool mY = solve_axis_track(y, lambda, trust, eps, dy);
            bool mZ = solve_axis_track(z, lambda, trust, eps, dz);
            bool mC = solve_axis_track(c, lambda, trust, eps, dc);
            if (mX) { warp_projection_by_step(x, dx); moved = true; }
            if (mY) { warp_projection_by_step(y, dy); moved = true; }
            if (mZ) { warp_projection_by_step(z, dz); moved = true; }
            if (mC) { warp_projection_by_step(c, dc); moved = true; }
            if (!moved) break;              // converged this frame
            any = true;
        }
        return any;
    }
    bool update_all_possible()
    {
        bool updated = false;
        for(auto &warp : warps) {
            if(warp.score > projection.score) {
                update_state(warp);
                updated = true;
            }
        }
        return updated;
    }

    bool update_from_max()
    {
        warp_bundle *best = &projection;
        for(auto &warp : warps)
            if(warp.score > best->score)
                best = &warp;

        if(best != &projection) {
            update_state(*best);
            return true;
        }
        return false;
    }

    bool update_heuristically()
    {
        bool updated = false;
        //best of x axis and roation around y (yaw)
        warp_bundle *best;
        best = &projection;
        if (warps[xp].score > best->score) best = &warps[xp];
        if (warps[xn].score > best->score) best = &warps[xn];
        if (warps[bp].score > best->score) best = &warps[bp];
        if (warps[bn].score > best->score) best = &warps[bn];
        if (warps[xp2].score > best->score) best = &warps[xp2];
        if (warps[xn2].score > best->score) best = &warps[xn2];
        if (warps[bp2].score > best->score) best = &warps[bp2];
        if (warps[bn2].score > best->score) best = &warps[bn2];
        if(best != &projection) update_state(*best);
        updated = updated || best != &projection;

        //best of y axis and rotation around x (pitch)
        best = &projection;
        if (warps[yp].score > best->score) best = &warps[yp];
        if (warps[yn].score > best->score) best = &warps[yn];
        if (warps[ap].score > best->score) best = &warps[ap];
        if (warps[an].score > best->score) best = &warps[an];
        if (warps[yp2].score > best->score) best = &warps[yp2];
        if (warps[yn2].score > best->score) best = &warps[yn2];
        if (warps[ap2].score > best->score) best = &warps[ap2];
        if (warps[an2].score > best->score) best = &warps[an2];
        if(best != &projection) update_state(*best);
        updated = updated || best != &projection;

        //best of roll
        best = &projection;
        if (warps[cp].score > best->score) best = &warps[cp];
        if (warps[cn].score > best->score) best = &warps[cn];
        if (warps[cp2].score > best->score) best = &warps[cp2];
        if (warps[cn2].score > best->score) best = &warps[cn2];
        if(best != &projection) update_state(*best);
        updated = updated || best != &projection;

        //best of z
        best = &projection;
        if (warps[zp].score > best->score) best = &warps[zp];
        if (warps[zn].score > best->score) best = &warps[zn];
        if (warps[zp2].score > best->score) best = &warps[zp2];
        if (warps[zn2].score > best->score) best = &warps[zn2];
        if(best != &projection) update_state(*best);
        updated = updated || best != &projection;

        return updated;
    }

    void score_overlay(double score, cv::Mat image)
    {
        if(score > cam[h]) score = cam[h];
        if(score < 0.0) score = 0.0;
        for(int i = 0; i < cam[w]*0.05; i++)
            for(int j = 0; j < (int)score; j++)
                image.at<float>(cam[h]-j-1, i) = 1.0;

    }

    cv::Mat create_translation_visualisation() 
    {
        static cv::Mat joined = cv::Mat::zeros(cam[h]*3, cam[w]*3, CV_32F);
        static cv::Mat joined_scaled = cv::Mat::zeros(cam[h], cam[w], CV_32F);
        cv::Mat tile;
        int col = 0; int row = 0;

        for(auto &warp : warps) {
            if(!warp.active) continue;
            switch(warp.axis) {
                case(x):
                    row = 1;
                    col = warp.delta > 0 ? 2 : 0;
                    break;
                case(y):
                    col = 1;
                    row = warp.delta > 0 ? 0 : 2;
                    break;
                case(z):
                    col = warp.delta > 0 ? 2 : 0;
                    row = warp.delta > 0 ? 2 : 0;
                    break;
                default:
                    continue;
            }
            tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
            cv::resize(warp.img_warp, tile, tile.size());
            score_overlay(warp.score, tile);
        }
        col = 1;
        row = 1;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(projection.img_warp, tile, tile.size());
        score_overlay(projection.score, tile);

        col = 0;
        row = 2;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(proc_obs, tile, tile.size());

        cv::resize(joined, joined_scaled, joined_scaled.size());

        return joined_scaled;

    }

    cv::Mat create_rotation_visualisation() 
    {
        static cv::Mat joined = cv::Mat::zeros(cam[h]*3, cam[w]*3, CV_32F);
        static cv::Mat joined_scaled = cv::Mat::zeros(cam[h], cam[w], CV_32F);
        cv::Mat tile;
        int col = 0; int row = 0;

        for(auto &warp : warps) {
            if(!warp.active) continue;
            switch(warp.axis) {
                case(b):
                    row = 1;
                    col = warp.delta > 0 ? 2 : 0;
                    break;
                case(a):
                    col = 1;
                    row = warp.delta > 0 ? 0 : 2;
                    break;
                case(c):
                    col = warp.delta > 0 ? 2 : 0;
                    row = warp.delta > 0 ? 2 : 0;
                    break;
                default:
                    continue;
            }
            tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
            cv::resize(warp.img_warp, tile, tile.size());
            score_overlay(warp.score, tile);
        }
        col = 1;
        row = 1;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(projection.img_warp, tile, tile.size());
        score_overlay(projection.score, tile);

        col = 0;
        row = 2;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(proc_obs, tile, tile.size());

        cv::resize(joined, joined_scaled, joined_scaled.size());

        return joined_scaled;

    }

};