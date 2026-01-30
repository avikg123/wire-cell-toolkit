#ifndef WIRECELLGEN_CORRELATEDADDNOISE_H
#define WIRECELLGEN_CORRELATEDADDNOISE_H

#include "WireCellAux/Logger.h"

#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/IFrameFilter.h"
#include "WireCellIface/IRandom.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace WireCell {
namespace Gen {

    class CorrelatedAddNoise : public Aux::Logger,
                              public IFrameFilter,
                              public IConfigurable
    {
      public:
        CorrelatedAddNoise();
        virtual ~CorrelatedAddNoise();

        // IConfigurable
        virtual WireCell::Configuration default_configuration() const override;
        virtual void configure(const WireCell::Configuration& cfg) override;

        // IFrameFilter
        virtual bool operator()(const input_pointer& inframe, output_pointer& outframe) override;

        // INamed is typically inherited through an interface in WCT.
        // We still must implement these if required by the base.
        virtual void set_name(const std::string& name) override;
        virtual std::string get_name() const override;

      private:
        void load_model(const std::string& fname);

        WireCell::ITrace::vector sorted_traces_by_channel(const WireCell::IFrame::pointer& frame) const;

        int    band_for_freq(double f) const;
        double interp_avg_mag_w(int w, double f) const;

        Eigen::MatrixXd make_correlated_noise(int nwires_frame, size_t nsamp) const;

        // ---- config ----
        std::string m_name{"CorrelatedAddNoise"};
        std::string m_model_file{"correlated_noise_model.json.bz2"};

        WireCell::IRandom::pointer m_rng;

        size_t m_nsamples{2128};
        double m_dt{0.5 * WireCell::units::us}; // WCT base time units (ns)
        double m_ifft_scale{1.0};               // should be 1.0 once FFT bin-doubling is correct

        // ---- model ----
        Eigen::VectorXd m_freq;                 // WCT frequency units (1/ns)
        Eigen::MatrixXd m_avg_mag;              // (wire, freqbin) mean |FFT| in WCT voltage units
        Eigen::VectorXi m_live_mask;            // 0/1 wires

        std::vector<Eigen::MatrixXd> m_A_band;

        struct BandInfo {
            double lo_f{0.0};
            double hi_f{0.0};
            size_t band_index{0};
            Eigen::VectorXd diagC;              // diag(A A^T) computed as rowwise sum of squares
        };
        std::vector<BandInfo> m_bands;

        mutable size_t m_count{0};
    };

} // namespace Gen
} // namespace WireCell

#endif // WIRECELLGEN_CORRELATEDADDNOISE_H