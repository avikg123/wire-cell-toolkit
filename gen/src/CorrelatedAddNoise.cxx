#include "WireCellGen/CorrelatedAddNoise.h"

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

static constexpr double SQRT_PI_OVER_2 = 0.5 * std::sqrt(M_PI);  // sqrt(pi)/2
static constexpr double SQRT_2_OVER_PI = std::sqrt(2.0 / M_PI);  // sqrt(2/pi)

inline bool ends_with(const std::string& s, const std::string& suf)
{
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline std::string slurp_file_text(const std::string& fname)
{
    std::ifstream in(fname, std::ios::in | std::ios::binary);
    if (!in.good()) {
        THROW(WireCell::IOError() << WireCell::errmsg{"CorrelatedAddNoise: cannot open model_file=" + fname});
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string slurp_file_bz2_text(const std::string& fname)
{
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) {
        THROW(WireCell::IOError() << WireCell::errmsg{"CorrelatedAddNoise: cannot open model_file=" + fname});
    }

    int bzerr = BZ_OK;
    BZFILE* bz = BZ2_bzReadOpen(&bzerr, fp, 0, 0, nullptr, 0);
    if (bzerr != BZ_OK || !bz) {
        std::fclose(fp);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{"CorrelatedAddNoise: BZ2_bzReadOpen failed for " + fname});
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
        THROW(WireCell::RuntimeError() << WireCell::errmsg{"CorrelatedAddNoise: BZ2_bzRead failed for " + fname});
    }

    BZ2_bzReadClose(&bzerr, bz);
    std::fclose(fp);
    return out;
}

inline Json::Value parse_json_string(const std::string& text, const std::string& label_for_errors)
{
    Json::CharReaderBuilder rb;
    rb["collectComments"] = false;

    Json::Value root;
    std::string errs;

    const std::unique_ptr<Json::CharReader> reader(rb.newCharReader());
    const bool ok = reader->parse(text.data(), text.data() + text.size(), &root, &errs);
    if (!ok) {
        THROW(WireCell::ValueError() << WireCell::errmsg{
            "CorrelatedAddNoise: JSON parse failed for " + label_for_errors + " : " + errs});
    }
    return root;
}

// FFTW c2r "numpy-like" irfft for one-sided spectrum
// Key point: with a one-sided spectrum, bins with conjugate partners must be doubled
// to match NumPy's irfft convention (this is why you previously needed ifft_scale=2).
inline void irfft_numpy_like_fftwf(const std::vector<std::complex<float>>& Xpos,
                                  std::vector<float>& x_time,
                                  int N,
                                  float final_scale)
{
    const int Kt = N / 2 + 1;
    if ((int)Xpos.size() != Kt) {
        THROW(WireCell::ValueError() << WireCell::errmsg{
            "irfft_numpy_like_fftwf: Xpos size != N/2+1"});
    }

    x_time.assign((size_t)N, 0.0f);

    fftwf_complex* in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)Kt);
    float*         out = (float*)fftwf_malloc(sizeof(float) * (size_t)N);

    if (!in || !out) {
        if (in)  fftwf_free(in);
        if (out) fftwf_free(out);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{
            "irfft_numpy_like_fftwf: fftwf_malloc failed"});
    }

    for (int k = 0; k < Kt; ++k) {
        in[k][0] = Xpos[(size_t)k].real();
        in[k][1] = Xpos[(size_t)k].imag();
    }

    const bool is_even = ((N % 2) == 0);
    if (is_even) {
        // Nyquist imag must be 0 for even N
        in[Kt - 1][1] = 0.0f;
    }

    // ---- FIX: factor-of-2 for bins with conjugate partners ----
    // Even N: double k=1..Kt-2.  Odd N: double k=1..Kt-1.
    const int k_max = is_even ? (Kt - 2) : (Kt - 1);
    for (int k = 1; k <= k_max; ++k) {
        in[k][0] *= 2.0f;
        in[k][1] *= 2.0f;
    }

    fftwf_plan plan = fftwf_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
    if (!plan) {
        fftwf_free(in);
        fftwf_free(out);
        THROW(WireCell::RuntimeError() << WireCell::errmsg{
            "irfft_numpy_like_fftwf: fftwf_plan_dft_c2r_1d failed"});
    }

    fftwf_execute(plan);

    // NumPy irfft normalization: 1/N
    const float invN = 1.0f / (float)N;
    for (int t = 0; t < N; ++t) {
        x_time[(size_t)t] = out[t] * invN * final_scale;
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

} // namespace

using namespace WireCell;
using namespace WireCell::Gen;

WIRECELL_FACTORY(CorrelatedAddNoise,
                 WireCell::Gen::CorrelatedAddNoise,
                 WireCell::INamed,
                 WireCell::IFrameFilter,
                 WireCell::IConfigurable)

CorrelatedAddNoise::CorrelatedAddNoise()
    : Aux::Logger("CorrelatedAddNoise")
{
}

CorrelatedAddNoise::~CorrelatedAddNoise() = default;

// ---- INamed ----
void CorrelatedAddNoise::set_name(const std::string& name)
{
    m_name = name;
}

std::string CorrelatedAddNoise::get_name() const
{
    return m_name;
}

// --------------------------------------------------------
Configuration CorrelatedAddNoise::default_configuration() const
{
    Configuration cfg;
    cfg["rng"]        = "Random:default";
    cfg["model_file"] = "correlated_noise_model.json.bz2";
    cfg["nsamples"]   = 2128;

    cfg["dt"]         = 0.5 * units::us; // 500 ns base units
    cfg["ifft_scale"] = 1.0;             // should now be correct

    return cfg;
}

// --------------------------------------------------------
void CorrelatedAddNoise::configure(const Configuration& cfg)
{
    auto rng_tn = get<std::string>(cfg, "rng", "Random:default");
    m_rng = Factory::find_tn<IRandom>(rng_tn);
    if (!m_rng) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: failed to get RNG tool"});
    }

    m_nsamples    = static_cast<size_t>(get<int>(cfg, "nsamples", 2128));
    m_dt          = get<double>(cfg, "dt", 0.5 * units::us);
    m_ifft_scale  = get<double>(cfg, "ifft_scale", 1.0);
    m_model_file  = get<std::string>(cfg, "model_file", "correlated_noise_model.json.bz2");

    // Accept dt either as WCT base units (ns-scale like 500) or seconds (<1).
    if (m_dt > 0.0 && m_dt < 1.0) {
        const double dt_seconds = m_dt;
        m_dt = dt_seconds * units::second;
        log->info("dt provided as seconds ({}) -> converted to WCT base time units ({} ns)", dt_seconds, m_dt);
    }

    load_model(m_model_file);

    if (m_avg_mag.cols() == 0) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: avg_mag has zero columns"});
    }
    if (m_avg_mag.cols() != m_freq.size()) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: avg_mag N_freq != freq.size()"});
    }

    log->info("Configured CorrelatedAddNoise: nsamples={} dt={}ns ifft_scale={} model_file={}",
              (int)m_nsamples, m_dt, m_ifft_scale, m_model_file);
}

// --------------------------------------------------------
void CorrelatedAddNoise::load_model(const std::string& fname)
{
    std::string text;
    if (ends_with(fname, ".bz2")) {
        text = slurp_file_bz2_text(fname);
    }
    else {
        text = slurp_file_text(fname);
    }

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
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: missing freq_hz or freq_ghz"});
    }

    const int Nf = static_cast<int>(jfreq.size());
    if (Nf <= 0) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: frequency array has zero length"});
    }

    m_freq.resize(Nf);
    for (int i = 0; i < Nf; ++i) {
        m_freq(i) = jfreq[(Json::ArrayIndex)i].asDouble() * freq_scale;
    }

    // ---- bands_khz ----
    if (!root.isMember("bands_khz")) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: missing bands_khz"});
    }
    const auto& jbands = root["bands_khz"];
    const int nbands = static_cast<int>(jbands.size());
    if (nbands <= 0) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: bands_khz is empty"});
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
            const size_t nt_model = static_cast<size_t>(meta["n_ticks"].asLargestUInt());
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

    // ---- avg_mag ----
    if (!root.isMember("avg_mag")) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: missing avg_mag"});
    }
    const auto& jam = root["avg_mag"];
    const int Nw = static_cast<int>(jam.size());
    if (Nw <= 0) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: avg_mag has zero rows"});
    }

    m_avg_mag.resize(Nw, Nf);
    for (int w = 0; w < Nw; ++w) {
        const auto& row = jam[(Json::ArrayIndex)w];
        if ((int)row.size() != Nf) {
            THROW(ValueError() << errmsg{
                "CorrelatedAddNoise: avg_mag row size mismatch (row="
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

    // ---- A_band ----
    if (!root.isMember("A_band")) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: missing A_band"});
    }
    const auto& jAb = root["A_band"];
    if ((int)jAb.size() != nbands) {
        THROW(ValueError() << errmsg{"CorrelatedAddNoise: A_band length != bands_khz length"});
    }

    m_A_band.clear();
    m_A_band.reserve(nbands);

    for (int b = 0; b < nbands; ++b) {
        const auto& jmat = jAb[(Json::ArrayIndex)b];
        if ((int)jmat.size() != Nw) {
            THROW(ValueError() << errmsg{"CorrelatedAddNoise: A_band[b] wrong N_wires"});
        }
        Eigen::MatrixXd A(Nw, Nw);
        for (int i = 0; i < Nw; ++i) {
            const auto& row = jmat[(Json::ArrayIndex)i];
            if ((int)row.size() != Nw) {
                THROW(ValueError() << errmsg{"CorrelatedAddNoise: A_band[b] row wrong N_wires"});
            }
            for (int j = 0; j < Nw; ++j) {
                A(i,j) = row[(Json::ArrayIndex)j].asDouble();
            }
        }
        m_A_band.push_back(std::move(A));
    }

    // ---- build band info ----
    m_bands.clear();
    m_bands.reserve(nbands);
    for (int b = 0; b < nbands; ++b) {
        const double lo_khz = jbands[(Json::ArrayIndex)b][0].asDouble();
        const double hi_khz = jbands[(Json::ArrayIndex)b][1].asDouble();

        BandInfo bi;
        bi.lo_f = lo_khz * units::kilohertz;
        bi.hi_f = hi_khz * units::kilohertz;
        bi.band_index = static_cast<size_t>(b);

        const Eigen::MatrixXd& A = m_A_band[(size_t)b];
        bi.diagC = A.array().square().rowwise().sum().matrix();
        for (int i = 0; i < bi.diagC.size(); ++i) {
            if (bi.diagC(i) < 1e-30) bi.diagC(i) = 1e-30;
        }

        m_bands.push_back(std::move(bi));
    }

    log->info("Loaded correlated noise model from {} with N_wires={}, N_freq={}, nbands={}",
              fname, (int)m_avg_mag.rows(), (int)m_freq.size(), (int)m_A_band.size());
}

// --------------------------------------------------------
ITrace::vector CorrelatedAddNoise::sorted_traces_by_channel(const IFrame::pointer& frame) const
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
inline int CorrelatedAddNoise::band_for_freq(double f) const
{
    for (size_t bi = 0; bi < m_bands.size(); ++bi) {
        if (f >= m_bands[bi].lo_f && f < m_bands[bi].hi_f) return (int)bi;
    }
    return -1;
}

// --------------------------------------------------------
inline double CorrelatedAddNoise::interp_avg_mag_w(int w, double f) const
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
Eigen::MatrixXd CorrelatedAddNoise::make_correlated_noise(int nwires_frame,
                                                         size_t nsamp) const
{
    const int N_wires_model = (int)m_avg_mag.rows();
    if (nwires_frame != N_wires_model) {
        log->warn("CorrelatedAddNoise: nwires_frame={} != model N_wires={}, truncating/minor mismatch",
                  nwires_frame, N_wires_model);
        nwires_frame = std::min(nwires_frame, N_wires_model);
    }

    const int    N   = (int)nsamp;
    const int    Kt  = N/2 + 1;
    const double dfT = 1.0 / (double(N) * m_dt); // 1/ns

    Eigen::MatrixXcd Xpos = Eigen::MatrixXcd::Zero(nwires_frame, Kt);
    if (Kt > 0) Xpos.col(0).setZero(); // DC

    const bool is_even = ((N % 2) == 0);
    const int nyq_k = is_even ? (Kt - 1) : -1;

    // ---- complex bins ----
    for (int kt = 1; kt < Kt; ++kt) {
        if (is_even && kt == nyq_k) continue;

        const double f = kt * dfT;
        const int b = band_for_freq(f);
        if (b < 0) continue;

        const auto& band = m_bands[(size_t)b];
        const Eigen::MatrixXd& A = m_A_band[band.band_index];
        const Eigen::VectorXd& diagC = band.diagC;

        Eigen::VectorXcd Z(N_wires_model);
        for (int w = 0; w < N_wires_model; ++w) {
            const double xr = m_rng->normal(0.0, 1.0);
            const double xi = m_rng->normal(0.0, 1.0);
            Z(w) = std::complex<double>(xr, xi) / std::sqrt(2.0);
        }

        const Eigen::VectorXcd U = A.cast<std::complex<double>>() * Z;

        for (int w = 0; w < nwires_frame; ++w) {
            if (m_live_mask(w) == 0) continue;

            const double target   = interp_avg_mag_w(w, f);
            const double baseline = SQRT_PI_OVER_2 * std::sqrt(diagC(w));
            const double scale    = (baseline > 0.0) ? (target / baseline) : 0.0;

            Xpos(w, kt) = U(w) * scale;
        }
    }

    // ---- Nyquist (real) if even ----
    if (is_even && nyq_k >= 0) {
        const double fnyq = nyq_k * dfT;
        const int b = band_for_freq(fnyq);
        if (b >= 0) {
            const auto& band = m_bands[(size_t)b];
            const Eigen::MatrixXd& A = m_A_band[band.band_index];
            const Eigen::VectorXd& diagC = band.diagC;

            Eigen::VectorXd nvec(N_wires_model);
            for (int w = 0; w < N_wires_model; ++w) {
                nvec(w) = m_rng->normal(0.0, 1.0);
            }
            const Eigen::VectorXd y = A * nvec;

            for (int w = 0; w < nwires_frame; ++w) {
                if (m_live_mask(w) == 0) continue;

                const double target   = interp_avg_mag_w(w, fnyq);
                const double baseline = SQRT_2_OVER_PI * std::sqrt(diagC(w));
                const double scale    = (baseline > 0.0) ? (target / baseline) : 0.0;

                Xpos(w, nyq_k) = std::complex<double>(y(w) * scale, 0.0);
            }
        }
    }

    // ---- IFFT via FFTW (Option A), explicit 1/N + bin-doubling ----
    Eigen::MatrixXd noise_time(nwires_frame, N);

    std::vector<std::complex<float>> one_sided((size_t)Kt);
    std::vector<float> x_time;

    for (int w = 0; w < nwires_frame; ++w) {
        for (int k = 0; k < Kt; ++k) {
            const auto xd = Xpos(w, k);
            one_sided[(size_t)k] = std::complex<float>((float)xd.real(), (float)xd.imag());
        }
        if (is_even) {
            one_sided[(size_t)(Kt - 1)] =
                std::complex<float>(one_sided[(size_t)(Kt - 1)].real(), 0.0f);
        }

        irfft_numpy_like_fftwf(one_sided, x_time, N, (float)m_ifft_scale);

        for (int t = 0; t < N; ++t) {
            noise_time(w, t) = (double)x_time[(size_t)t];
        }
    }

    return noise_time;
}

// --------------------------------------------------------
bool CorrelatedAddNoise::operator()(const input_pointer& inframe,
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

    const size_t ncharge = traces[0]->charge().size();
    if (ncharge != m_nsamples) {
        log->warn("CorrelatedAddNoise: frame nsamples={} differs from configured nsamples={}, adapting per-frame",
                  ncharge, m_nsamples);
    }

    Eigen::MatrixXd noise = make_correlated_noise((int)nwires_frame, ncharge);

    ITrace::vector outtraces;
    outtraces.reserve(nwires_frame);

    for (size_t i = 0; i < nwires_frame; ++i) {
        const auto& intrace = traces[i];
        const int  chid     = intrace->channel();
        auto charge         = intrace->charge();

        if ((int)i < noise.rows()) {
            for (size_t t = 0; t < charge.size(); ++t) {
                charge[t] += noise((int)i, (int)t);
            }
        }

        auto trace = std::make_shared<Aux::SimpleTrace>(chid, intrace->tbin(), charge);
        outtraces.push_back(trace);
    }

    outframe = std::make_shared<Aux::SimpleFrame>(inframe->ident(),
                                                  inframe->time(),
                                                  outtraces,
                                                  inframe->tick());

    log->debug("CorrelatedAddNoise: call={} frame={} {} traces",
               m_count, inframe->ident(), outtraces.size());
    ++m_count;
    return true;
}