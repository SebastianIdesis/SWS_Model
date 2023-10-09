function [ params ] = DefaultParams_Idesis_G1_delta(varargin)
%%DEFAULTPARAMS Default parameter settings for DMF simulation
%
%   P = DEFAULTPARAMS() yields a struct with default values for all necessary
%   parameters for the DMF model. The default structural connectivity is the
%   DTI fiber consensus matrix obtained from the HCP dataset, using a
%   Schaeffer-100 parcellation.
%
%   P = DEFAULTPARAMS('key', 'value', ...) adds (or replaces) field 'key' in
%   P with given value.
%
% Pedro Mediano, Feb 2021

params = [];

% Connectivity matrix
if any(strcmp(varargin, 'C'))
  C = [];
else
  try
    p = strrep(mfilename('fullpath'), 'DefaultParams', '');
    C = dlmread([p, '../data/DTI_fiber_consensus_HCP.csv'], ',');
    C = C/max(C(:));
  catch
    error('No connectivity matrix provided, and default matrix not found.');
  end

end


% DMF parameters
params.C         = C;
%load('atlas_Yeo_200_7.trk.gz.Schaefer2018_200Parcels_7Networks_order_plus_subcort.count.end.connectivity.mat')
%params.C         = connectivity;
params.receptors = 0;
params.dt        = 0.1;     % ms
params.taon      = 100;     % NMDA tau ms
params.tauad     = 100;     % NMDA tau ms
params.taog      = 10;      % GABA tau ms
params.gamma     = 0.641;   % Kinetic Parameter of Excitation
params.sigma     = 0.01;    % Noise SD nA
params.JN        = 0.15;    % excitatory synaptic coupling nA
params.I0        = 0.382;   % effective external input nA
params.Iext      = 0.382; % ADDED TO SIMULATE EXTERNAL STIMULATION ORIGINAL = 0.382
params.Jexte     = 1.;      % external->E coupling
params.Jexti     = 0.7;     % external->I coupling
params.w         = 1.4;     % local excitatory recurrence
params.de        = 0.16;    % excitatory non linear shape parameter
params.Ie        = 125/310; % excitatory threshold for nonlineariy
params.g_e       = 310.;    % excitatory conductance
params.di        = 0.087;   % inhibitory non linear shape parameter
params.Ii        = 177/615; % inhibitory threshold for nonlineariy
params.g_i       = 615.;    % inhibitory conductance
params.wgaine    = 0;       % neuromodulatory gain
params.wgaini    = 0;       % neuromodulatory gain
params.G         = 1;       % Global Coupling Parameter % CHECK THIS Range 0 a 3 en steps 0.01


% Balloon-Windkessel parameters (from firing rates to BOLD signal)
params.TR  = 2;     % number of seconds to sample bold signal
params.dtt = 0.001; % BW integration step, in seconds

% Parallel computation parameters
params.batch_size = 5000;

% Add/replace remaining parameters
for i=1:2:length(varargin)
  params.(varargin{i}) = varargin{i+1};
end

% If feedback inhibitory control not provided, use heuristic
if ~any(strcmp(varargin, 'J'))
  params.J = 0.75*params.G*sum(params.C, 1)' + 1; % este .75 seria el alpha
end

end

