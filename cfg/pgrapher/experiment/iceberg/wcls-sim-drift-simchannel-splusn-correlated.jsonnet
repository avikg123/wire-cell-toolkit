// wcls-sim-drift-simchannel-splusn-correlated.jsonnet
//
// Replace old GroupNoiseModel/IncoherentAddNoise with CorrelatedAddNoise.
// Assumes noise model stores:
//   - freq_ghz : GHz (== 1/ns in WCT base time units)
//   - avg_mag  : MV  (WCT base voltage)
//
// And CorrelatedAddNoise adds noise in MV to frames in MV.

local g  = import 'pgraph.jsonnet';
local f  = import 'pgrapher/common/funcs.jsonnet';
local wc = import 'wirecell.jsonnet';

local io           = import 'pgrapher/common/fileio.jsonnet';
local tools_maker  = import 'pgrapher/common/tools.jsonnet';
local params_maker = import 'pgrapher/experiment/iceberg/simparams.jsonnet';

// -----------------------------------------------------------------------------
// FHiCL-provided externals
// -----------------------------------------------------------------------------
local fcl_params = {
  G4RefTime: std.extVar('G4RefTime') * wc.us,
};

local params = params_maker(fcl_params) {
  lar: super.lar {
    DL: std.extVar('DL') * wc.cm2 / wc.s,
    DT: std.extVar('DT') * wc.cm2 / wc.s,
    lifetime: std.extVar('lifetime') * wc.ms,
    drift_speed: std.extVar('driftSpeed') * wc.mm / wc.us,
  },
};

local tools = tools_maker(params);

local sim_maker = import 'pgrapher/experiment/iceberg/sim.jsonnet';
local sim = sim_maker(params, tools);

local nanodes = std.length(tools.anodes);
local anode_iota = std.range(0, nanodes - 1);

// -----------------------------------------------------------------------------
// HARD-CODE correlated noise model path (MV + GHz model)
// -----------------------------------------------------------------------------
local corr_model = "/exp/dune/app/users/avighosh/noise_model/model_files/correlated_noise_model_WCT_MV_GHz.json.bz2";

// -----------------------------------------------------------------------------
// WCLS input
// -----------------------------------------------------------------------------
local wcls_maker = import 'pgrapher/ui/wcls/nodes.jsonnet';
local wcls = wcls_maker(params, tools);

local wcls_input = {
  depos: wcls.input.depos(
    name='',
    art_tag='largeant:LArG4DetectorServicevolTPCActive'
  ),
};

// -----------------------------------------------------------------------------
// MegaAnode plane
// -----------------------------------------------------------------------------
local mega_anode = {
  type: 'MegaAnodePlane',
  name: 'meganodes',
  data: {
    anodes_tn: [wc.tn(anode) for anode in tools.anodes],
  },
};

// -----------------------------------------------------------------------------
// Digitization scale
// NOTE: Despite variable name, this is effectively "ADC per WCT-voltage-unit".
// params.adc.fullscale is in WCT voltage units (MV base).
// -----------------------------------------------------------------------------
local resolution = params.adc.resolution;
local fullscale  = params.adc.fullscale[1] - params.adc.fullscale[0];
local ADC_per_Vwct = ((1 << resolution) - 1) / fullscale;

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------
local wcls_output = {

  // signal waveform from simulation
  sim_signals: g.pnode({
    type: 'wclsFrameSaver',
    name: 'simsignals',
    data: {
      anode: wc.tn(mega_anode),
      digitize: true,
      frame_tags: ['sig'],
      frame_scale: [ADC_per_Vwct],   // convert WCT voltage (MV base) -> ADC counts
    },
  }, nin=1, nout=1, uses=[mega_anode]),

  // ADC output from simulation
  sim_digits: g.pnode({
    type: 'wclsFrameSaver',
    name: 'simdigits',
    data: {
      anode: wc.tn(mega_anode),
      digitize: true,
      frame_tags: ['daq'],
      pedestal_mean: 'native',
    },
  }, nin=1, nout=1, uses=[mega_anode]),

  nf_digits: wcls.output.digits(name='nfdigits', tags=['raw']),
  sp_signals: wcls.output.signals(name='spsignals', tags=['gauss', 'wiener']),
  sp_thresholds: wcls.output.thresholds(name='spthresholds', tags=['threshold']),
};

// -----------------------------------------------------------------------------
// Core sim chain
// -----------------------------------------------------------------------------
local drifter = sim.drifter;
local bagger  = sim.make_bagger();
local signal_pipes = sim.signal_pipelines;

local rng = tools.random;

local wcls_simchannel_sink = g.pnode({
  type: 'wclsSimChannelSink',
  name: 'postdrift',
  data: {
    artlabel: 'simpleSC',
    anodes_tn: [wc.tn(anode) for anode in tools.anodes],
    rng: wc.tn(rng),
    tick: 0.5 * wc.us,
    start_time: -0.25 * wc.ms,
    readout_time: self.tick * 6000,
    nsigma: 3.0,
    drift_speed: params.lar.drift_speed,
    u_to_rp: 100 * wc.mm,
    v_to_rp: 100 * wc.mm,
    y_to_rp: 100 * wc.mm,
    u_time_offset: 0.0 * wc.us,
    v_time_offset: 0.0 * wc.us,
    y_time_offset: 0.0 * wc.us,
    g4_ref_time: fcl_params.G4RefTime,
    use_energy: true,
  },
}, nin=1, nout=1, uses=tools.anodes);

// fan out anodes, simulate signals
local multipass = [
  g.pipeline([ signal_pipes[n] ], 'multipass%d' % n)
  for n in anode_iota
];

local outtags = ['orig%d' % n for n in anode_iota];
local bi_manifold = f.fanpipe('DepoSetFanout', multipass, 'FrameFanin', 'sn_mag_nf', outtags);

// merge to a single tag "sig"
local retagger = g.pnode({
  type: 'Retagger',
  data: {
    tag_rules: [{
      frame: { '.*': 'orig' },
      merge: { 'orig\\d+': 'sig' },
    }],
  },
}, nin=1, nout=1);

// -----------------------------------------------------------------------------
// Correlated noise injection branch
// -----------------------------------------------------------------------------

// select per-anode channel ranges from the 'sig' frame
local chsel_pipes = [
  g.pnode({
    type: 'ChannelSelector',
    name: 'chsel%d' % n,
    data: {
      channels: std.range(1280 * n, 1280 * (n + 1) - 1),
      tags: ['sig'],
    },
  }, nin=1, nout=1)
  for n in anode_iota
];

// one correlated noise node per lane
local corrnoises = [
  g.pnode({
    type: 'CorrelatedAddNoise',
    name: 'corrnoise%d' % n,
    data: {
      rng: wc.tn(tools.random),
      model_file: corr_model,

      // These are in WCT base units already:
      nsamples: params.daq.nticks,
      dt: params.daq.tick,       // ns (WCT base time units)

      // optional: keep at 1.0 unless you need a quick normalization tweak
      ifft_scale: 1.0,
    },
  }, nin=1, nout=1, uses=[tools.random])
  for n in anode_iota
];

// digitize noisy frames into splusnN
local digitizers_noise = [
  sim.digitizer(tools.anodes[n], name='digitizer-noise%d' % n, tag='splusn%d' % n)
  for n in anode_iota
];

// per-lane pipeline: select -> add noise -> digitize
local multipass_noise = [
  g.pipeline([
    chsel_pipes[n],
    corrnoises[n],
    digitizers_noise[n],
  ], 'multipass-noise%d' % n)
  for n in anode_iota
];

// fanout/fanin to collect the noisy digitized lanes
local outtags_noise = ['splusn%d' % n for n in anode_iota];
local bi_manifold_noise =
  f.fanpipe('FrameFanout', multipass_noise, 'FrameFanin', 'noisedigit', outtags_noise);

// merge to final "daq"
local retagger_noise = g.pnode({
  type: 'Retagger',
  name: 'retagger-noise',
  data: {
    tag_rules: [{
      frame: { '.*': 'splusn' },
      merge: { 'splusn\\d+': 'daq' },
    }],
  },
}, nin=1, nout=1);

// -----------------------------------------------------------------------------
// Sink + app wrapper
// -----------------------------------------------------------------------------
local sink = sim.frame_sink;

local graph = g.pipeline([
  wcls_input.depos,
  drifter,
  wcls_simchannel_sink,
  bagger,

  // signal path
  bi_manifold,
  retagger,
  wcls_output.sim_signals,

  // noise+digitize path
  bi_manifold_noise,
  retagger_noise,
  wcls_output.sim_digits,

  sink
]);

local app = {
  type: 'Pgrapher',
  data: { edges: g.edges(graph) },
};

g.uses(graph) + [app]