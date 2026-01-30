#ifndef WIRECELLGEN_UNCORRELATEDADDNOISE_H
#define WIRECELLGEN_UNCORRELATEDADDNOISE_H

#include "WireCellAux/Logger.h"
#include "WireCellIface/IFrameFilter.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/IRandom.h"

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace WireCell {
namespace Gen {

    class UncorrelatedAddNoise : public Aux::Logger,
                                public IFrameFilter,
                                public IConfigurable
    {
      public:
        UncorrelatedAddNoise();
        virtual ~UncorrelatedAddNoise();

        // IConfigurable
        virtual Configuration default_configuration() const override;
        virtual void configure(const Configuration& cfg) override;

        // IFrameFilter
        virtual bool operator()(const input_pointer& inframe,
                                output_pointer& outframe) override;

        // INamed (comes via INode in the IFrameFilter chain)
        virtual void set_name(const std::string& name) override;
        virtual std::string get_name() const override;

      private:
        void load_model(const std::string& fname);

        ITrace::vector sorted_traces_by_channel(const IFrame::pointer& frame) const;

        // avg_mag interpolation onto target f (in WCT base freq units == 1/ns)
        double interp_avg_mag_w(int w, double f) const;

      private:
        std::string m_name{"UncorrelatedAddNoise"};

        IRandom::pointer m_rng{nullptr};

        size_t m_nsamples{2128};
        double m_dt{0.5 * units::us};     // WCT base time units (ns)
        std::string m_model_file{"uncorrelated_noise_model.json.bz2"};

        // model
        Eigen::VectorXd m_freq;           // in WCT base freq units (1/ns)
        Eigen::MatrixXd m_avg_mag;        // [wire, freqbin] in WCT base MV
        Eigen::VectorXi m_live_mask;      // 0/1 per wire

        size_t m_count{0};
    };

} // namespace Gen
} // namespace WireCell

#endif