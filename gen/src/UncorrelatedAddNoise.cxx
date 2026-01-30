#include "WireCellGen/UncorrelatedAddNoise.h"

#include "WireCellAux/SimpleTrace.h"
#include "WireCellAux/SimpleFrame.h"

#include "WireCellUtil/Configuration.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Units.h"

#include <json/json.h>
#include <fftw3.h>
#include <bzlib.h>

#include <algorithm>
#include <complex>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <memory>

namespace {

static constexpr double SQRT_PI_OVER_2 = 0.5 * std::sqrt(M_PI); // Rayleigh mean for CN(0,1)
static constexpr double SQRT_2_OVER_PI = std::sqrt(2.0 / M_PI); // mean(|N(0,1)|)

// This factor is the "clean" replacement for jsonnet ifft_scale.
// Empirically, to match a model built with NumPy's rfft(|.|) convention,
// FFTW c2r + (1/N) lands ~0.5 low in |rfft|. Multiplying the time-domain
// result by 2 fixes the convention mismatch (same as your correlated case).
static constexpr float IFFT_EXTRA_SCALE = 2.0f;

// -----------------------------
// file helpers (plain + bz2)
// -----------------------------
inline bool ends_with(const std::string& s, const std::string& suf)
{
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline std::string slurp_file_text(const std::string& fname)
{
    std::ifstream in(fname, std::ios::in | std::ios::binary);
    if (!in.good()) {
        THROW(WireCell::IOError() << WireCell::errmsg{"UncorrelatedAddNoise: cannot open model_file=" + fname});
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string slurp_file_bz2_text(const std::string& fname)
{
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) {
        THROW(WireCell::IOError() << WireCell::errmsg{"UncorrelatedAddNoise: cannot open model_file=" + fname});
    }

    int bzerr = BZ_OK;
    BZFILE* bz = BZ2_bzReadOpen(&bzerr, fp, 0, 0, nullptr, 0);
    if (bzerr != BZ_OK || !bz) {
        std::fclose(fp);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{"UncorrelatedAddNoise: BZ2_bzReadOpen failed for " + fname});
    }

    std::string out;
    out.reserve(1024 * 1024);

    const int CHUNK = 1 << 15;
    std::vector<char> buf((size_t)CHUNK);

    while (true) {
        int nread = BZ2_bzRead(&bzerr, bz, buf.data(), CHUNK);
        if (bzerr == BZ_OK || bzerr == BZ_STREAM_END) {
            if (nread > 0) out.append(buf.data(), (size_t)nread);
            if (bzerr == BZ_STREAM_END) break;
            continue;
        }
        BZ2_bzReadClose(&bzerr, bz);
        std::fclose(fp);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{"UncorrelatedAddNoise: BZ2_bzRead failed for " + fname});
    }

    BZ2_bzReadClose(&bzerr, bz);
    std::fclose(fp);
    return out;
}

inline Json::Value parse_json_string(const std::string& text, const std::string& label)
{
    Json::CharReaderBuilder rb;
    rb["collectComments"] = false;

    Json::Value root;
    std::string errs;

    const std::unique_ptr<Json::CharReader> reader(rb.newCharReader());
    const bool ok = reader->parse(text.data(), text.data() + text.size(), &root, &errs);
    if (!ok) {
        THROW(WireCell::ValueError() << WireCell::errmsg{
            "UncorrelatedAddNoise: JSON parse failed for " + label + " : " + errs});
    }
    return root;
}

// -----------------------------
// numpy-like irfft via FFTW c2r:
// - FFTW c2r is unnormalized
// - NumPy irfft has 1/N normalization
// - plus we apply an extra factor 2 to match the model's avg_mag convention
//   (same fix you needed for correlated).
// -----------------------------
inline void irfft_numpy_like_fftwf_times2(const std::vector<std::complex<float>>& Xpos,
                                         std::vector<float>& x_time,
                                         int N)
{
    const int Kt = N / 2 + 1;
    if ((int)Xpos.size() != Kt) {
        THROW(WireCell::ValueError() << WireCell::errmsg{
            "irfft_numpy_like_fftwf_times2: Xpos size != N/2+1"});
    }

    x_time.assign((size_t)N, 0.0f);

    fftwf_complex* in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)Kt);
    float*         out = (float*)fftwf_malloc(sizeof(float) * (size_t)N);

    if (!in || !out) {
        if (in)  fftwf_free(in);
        if (out) fftwf_free(out);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{
            "irfft_numpy_like_fftwf_times2: fftwf_malloc failed"});
    }

    for (int k = 0; k < Kt; ++k) {
        in[k][0] = Xpos[(size_t)k].real();
        in[k][1] = Xpos[(size_t)k].imag();
    }
    if ((N % 2) == 0) {
        in[Kt - 1][1] = 0.0f; // Nyquist imag must be zero
    }

    fftwf_plan plan = fftwf_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
    if (!plan) {
        fftwf_free(in);
        fftwf_free(out);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{
            "irfft_numpy_like_fftwf_times2: fftwf_plan_dft_c2r_1d failed"});
    }

    fftwf_execute(plan);

    const float scale = IFFT_EXTRA_SCALE / (float)N;  // (2/N)
    for (int t = 0; t < N; ++t) {
        x_time[(size_t)t] = out[t] * scale;
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

} // namespace

using namespace WireCell;
using namespace WireCell::Gen;

WIRECELL_FACTORY(UncorrelatedAddNoise,
                 WireCell::Gen::UncorrelatedAddNoise,
                 WireCell::IFrameFilter,
                 WireCell::IConfigurable)

UncorrelatedAddNoise::UncorrelatedAddNoise()
    : Aux::Logger("UncorrelatedAddNoise")
{
}

UncorrelatedAddNoise::~UncorrelatedAddNoise() = default;

// ---- INamed ----
void UncorrelatedAddNoise::set_name(const std::string& name)
{
    m_name = name;
}

std::string UncorrelatedAddNoise::get_name() const
{
    return m_name;
}

// --------------------------------------------------------
Configuration UncorrelatedAddNoise::default_configuration() const
{
    Configuration cfg;
    cfg["rng"]        = "Random:default";
    cfg["model_file"] = "uncorrelated_noise_model.json.bz2";
    cfg["nsamples"]   = 2128;
    cfg["dt"]         = 0.5 * units::us; // 500 ns (WCT base units)
    return cfg;
}

// --------------------------------------------------------
void UncorrelatedAddNoise::configure(const Configuration& cfg)
{
    auto rng_tn = get<std::string>(cfg, "rng", "Random:default");
    m_rng = Factory::find_tn<IRandom>(rng_tn);
    if (!m_rng) {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: failed to get RNG tool"});
    }

    m_nsamples    = static_cast<size_t>(get<int>(cfg, "nsamples", 2128));
    m_dt          = get<double>(cfg, "dt", 0.5 * units::us);
    m_model_file  = get<std::string>(cfg, "model_file", "uncorrelated_noise_model.json.bz2");

    // Accept dt either as WCT base (ns-scale like 500) or seconds (<1)
    if (m_dt > 0.0 && m_dt < 1.0) {
        const double dt_seconds = m_dt;
        m_dt = dt_seconds * units::second;
        log->info("dt provided as seconds ({}) -> converted to WCT base time units ({} ns)", dt_seconds, m_dt);
    }

    load_model(m_model_file);

    if (m_avg_mag.cols() == 0) {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: avg_mag has zero columns"});
    }
    if (m_avg_mag.cols() != m_freq.size()) {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: avg_mag N_freq != freq.size()"});
    }

    log->info("Configured UncorrelatedAddNoise: nsamples={} dt={}ns model_file={} (internal IFFT extra scale = {})",
              (int)m_nsamples, m_dt, m_model_file, IFFT_EXTRA_SCALE);
}

// --------------------------------------------------------
void UncorrelatedAddNoise::load_model(const std::string& fname)
{
    std::string text = ends_with(fname, ".bz2") ? slurp_file_bz2_text(fname) : slurp_file_text(fname);
    Json::Value root = parse_json_string(text, fname);

    // ---- frequency axis ----
    Json::Value jfreq;
    double freq_scale = 1.0;

    if (root.isMember("freq_hz")) {
        jfreq = root["freq_hz"];
        freq_scale = units::hertz;
    }
    else if (root.isMember("freq_ghz")) {
        jfreq = root["freq_ghz"];
        freq_scale = (1.0e9 * units::hertz);
    }
    else {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: missing freq_hz or freq_ghz"});
    }

    const int Nf = (int)jfreq.size();
    if (Nf <= 0) {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: frequency array has zero length"});
    }

    m_freq.resize(Nf);
    for (int i = 0; i < Nf; ++i) {
        m_freq(i) = jfreq[(Json::ArrayIndex)i].asDouble() * freq_scale; // WCT base freq (1/ns)
    }

    // ---- avg_mag units ----
    double mag_scale = 1.0; // assume MV base

    if (root.isMember("meta")) {
        const auto& meta = root["meta"];

        if (meta.isMember("dt_ns")) {
            const double dt_model = meta["dt_ns"].asDouble();
            if (std::fabs(dt_model - m_dt) > 1e-6) {
                log->warn("Model meta.dt_ns={} differs from configured dt={} (both expected WCT base ns)",
                          dt_model, m_dt);
            }
        }
        if (meta.isMember("n_ticks")) {
            const size_t nt_model = (size_t)meta["n_ticks"].asLargestUInt();
            if (nt_model != m_nsamples) {
                log->warn("Model meta.n_ticks={} differs from configured nsamples={}", nt_model, m_nsamples);
            }
        }
        if (meta.isMember("stored_units") && meta["stored_units"].isMember("avg_mag")) {
            const std::string u = meta["stored_units"]["avg_mag"].asString();
            if (u == "MV")      mag_scale = units::megavolt;
            else if (u == "mV") mag_scale = units::millivolt;
            else {
                log->warn("Unknown meta.stored_units.avg_mag='{}' -> assuming MV base", u);
                mag_scale = 1.0;
            }
        }
    }

    if (!root.isMember("avg_mag")) {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: missing avg_mag"});
    }

    const auto& jam = root["avg_mag"];
    const int Nw = (int)jam.size();
    if (Nw <= 0) {
        THROW(ValueError() << errmsg{"UncorrelatedAddNoise: avg_mag has zero rows"});
    }

    m_avg_mag.resize(Nw, Nf);
    for (int w = 0; w < Nw; ++w) {
        const auto& row = jam[(Json::ArrayIndex)w];
        if ((int)row.size() != Nf) {
            THROW(ValueError() << errmsg{
                "UncorrelatedAddNoise: avg_mag row size mismatch (row="
                + std::to_string((int)row.size()) + " != N_freq=" + std::to_string(Nf) + ")"
            });
        }
        for (int k = 0; k < Nf; ++k) {
            m_avg_mag(w, k) = row[(Json::ArrayIndex)k].asDouble() * mag_scale;
        }
    }

    // ---- live_mask ----
    m_live_mask.resize(Nw);
    m_live_mask.setOnes();
    if (root.isMember("live_mask")) {
        const auto& jlive = root["live_mask"];
        if ((int)jlive.size() == Nw) {
            for (int w = 0; w < Nw; ++w) m_live_mask(w) = jlive[(Json::ArrayIndex)w].asInt();
        }
        else {
            log->warn("live_mask size {} != N_wires {}, ignoring", (int)jlive.size(), Nw);
        }
    }

    log->info("Loaded uncorrelated noise model from {} with N_wires={}, N_freq={}",
              fname, (int)m_avg_mag.rows(), (int)m_freq.size());
}

// --------------------------------------------------------
ITrace::vector UncorrelatedAddNoise::sorted_traces_by_channel(const IFrame::pointer& frame) const
{
    ITrace::vector out;
    auto tvp = frame->traces();
    if (tvp && !tvp->empty()) {
        out = *tvp;
        std::sort(out.begin(), out.end(),
                  [](const ITrace::pointer& a, const ITrace::pointer& b) {
                      return a->channel() < b->channel();
                  });
    }
    return out;
}

// --------------------------------------------------------
double UncorrelatedAddNoise::interp_avg_mag_w(int w, double f) const
{
    const int Nf = (int)m_freq.size();
    if (Nf <= 1) return m_avg_mag(w, 0);

    if (f <= m_freq(0)) return m_avg_mag(w, 0);
    if (f >= m_freq(Nf-1)) return m_avg_mag(w, Nf-1);

    const double* begin = m_freq.data();
    const double* end   = m_freq.data() + Nf;
    const double* it = std::lower_bound(begin, end, f);

    int i1 = (int)(it - begin);
    if (i1 <= 0) return m_avg_mag(w, 0);
    if (i1 >= Nf) return m_avg_mag(w, Nf-1);
    int i0 = i1 - 1;

    const double f0 = m_freq(i0);
    const double f1 = m_freq(i1);
    const double a  = (f1 > f0) ? ((f - f0) / (f1 - f0)) : 0.0;

    return (1.0 - a) * m_avg_mag(w, i0) + a * m_avg_mag(w, i1);
}

// --------------------------------------------------------
bool UncorrelatedAddNoise::operator()(const input_pointer& inframe,
                                      output_pointer& outframe)
{
    if (!inframe) {
        outframe = nullptr;
        log->debug("EOS at call={}", m_count);
        ++m_count;
        return true;
    }

    auto traces = sorted_traces_by_channel(inframe);
    const size_t nwires_frame = traces.size();

    if (nwires_frame == 0) {
        outframe = inframe;
        ++m_count;
        return true;
    }

    const size_t nsamp = traces[0]->charge().size();
    if (nsamp != m_nsamples) {
        log->warn("UncorrelatedAddNoise: frame nsamples={} differs from configured nsamples={}, adapting per-frame",
                  nsamp, m_nsamples);
    }

    const int N  = (int)nsamp;
    const int Kt = N/2 + 1;
    const bool is_even = ((N % 2) == 0);
    const int nyq_k = is_even ? (Kt - 1) : -1;

    const int N_wires_model = (int)m_avg_mag.rows();
    const int nw_use = std::min((int)nwires_frame, N_wires_model);

    const double dfT = 1.0 / (double(N) * m_dt); // 1/ns

    ITrace::vector outtraces;
    outtraces.reserve(nwires_frame);

    std::vector<std::complex<float>> Xpos((size_t)Kt);
    std::vector<float> x_time;

    for (int i = 0; i < (int)nwires_frame; ++i) {
        const auto& intrace = traces[(size_t)i];
        const int chid = intrace->channel();
        auto charge = intrace->charge();

        if (i < nw_use && m_live_mask(i) != 0) {

            // DC
            Xpos[0] = std::complex<float>(0.0f, 0.0f);

            // complex positive freqs (exclude Nyquist for even)
            for (int k = 1; k < Kt; ++k) {
                if (is_even && k == nyq_k) continue;

                const double f = k * dfT;
                const double target = interp_avg_mag_w(i, f);

                // Z ~ CN(0,1) => E|Z| = sqrt(pi)/2
                const double xr = m_rng->normal(0.0, 1.0);
                const double xi = m_rng->normal(0.0, 1.0);
                const std::complex<double> Z = std::complex<double>(xr, xi) / std::sqrt(2.0);

                const double scale = (SQRT_PI_OVER_2 > 0.0) ? (target / SQRT_PI_OVER_2) : 0.0;
                const std::complex<double> Xk = Z * scale;

                Xpos[(size_t)k] = std::complex<float>((float)Xk.real(), (float)Xk.imag());
            }

            // Nyquist (real) for even N
            if (is_even && nyq_k >= 0) {
                const double fnyq = nyq_k * dfT;
                const double target = interp_avg_mag_w(i, fnyq);

                const double n = m_rng->normal(0.0, 1.0);
                // E|N(0,1)| = sqrt(2/pi)
                const double scale = (SQRT_2_OVER_PI > 0.0) ? (target / SQRT_2_OVER_PI) : 0.0;

                Xpos[(size_t)nyq_k] = std::complex<float>((float)(n * scale), 0.0f);
            }

            // IFFT: FFTW c2r, then (2/N) to match NumPy-built model convention
            irfft_numpy_like_fftwf_times2(Xpos, x_time, N);

            for (int t = 0; t < N; ++t) {
                charge[(size_t)t] += (double)x_time[(size_t)t];
            }
        }

        outtraces.push_back(std::make_shared<Aux::SimpleTrace>(chid, intrace->tbin(), charge));
    }

    outframe = std::make_shared<Aux::SimpleFrame>(inframe->ident(),
                                                  inframe->time(),
                                                  outtraces,
                                                  inframe->tick());

    log->debug("UncorrelatedAddNoise: call={} frame={} {} traces",
               m_count, inframe->ident(), outtraces.size());
    ++m_count;
    return true;
}