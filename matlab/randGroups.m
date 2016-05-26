function p_values = randGroups(tablefile, varargin)

% p_values = randGroups(tablefile, varargin)
% Randomize mean effects between two or more groups.
% Returns bootstrapped group-wise p-values and plots (if varargin{1} =
% true)
%
%  Inputs:
%      tablefile - a string (path to a .csv file) or a table of format:
%                     group     'LP'       'PD'       'GM'
%                    values     0.814      0.345      1.113
%                               ...        ...        ...
%      plot      - varargin{1}, defaults to true, simple summary plot
%      samples   - varargin{2}, number of shuffles, defaults to 1000 or
%                  however many combinations are possible (if <1000)
%
%  Outputs:
%      p_values  - bootstrapped p-values returned by groups