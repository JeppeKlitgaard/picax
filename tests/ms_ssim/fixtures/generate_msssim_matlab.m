function generate_msssim_matlab(msssim_dir)
% Generate msssim_matlab.mat, the MS-SSIM values computed by Wang et al.'s
% reference implementation, for the parity tests of picax's ms_ssim.
%
% The reference implementation is not redistributed here. Download it from
%
%   https://ece.uwaterloo.ca/~z70wang/research/ssim/msssim.zip
%   (sha256 ff1426726ffa50892627407da64141d5ca0e16d48c80eb38a74d63487606602e)
%
% extract it, and run
%
%   generate_msssim_matlab('/path/to/extracted/msssim')
%
% Requires the Image Processing Toolbox. msssim.m hardcodes a dynamic range of
% 255, so every input is uint8.

addpath(msssim_dir);

rng(0, 'twister');

wang_weights = [0.0448 0.2856 0.3001 0.2363 0.1333];
default_K = [0.01 0.03];

cases = struct('name', {}, 'x', {}, 'y', {}, 'K', {}, 'level', {}, ...
    'weight', {}, 'expected', {});

% Even extents, at the minimum size msssim.m accepts for five scales.
x = make_image(176, 192);
cases(end + 1) = make_case('noise-even', x, add_noise(x, 10), ...
    default_K, 5, wang_weights);

% Odd extents exercise the symmetric padding of the downsampling filter.
x = make_image(181, 197);
cases(end + 1) = make_case('blur-odd', x, ...
    uint8(imgaussfilt(double(x), 1.5)), default_K, 5, wang_weights);

% A luminance and contrast change leaves structure intact.
x = make_image(193, 179);
cases(end + 1) = make_case('contrast-brightness', x, ...
    uint8(0.7 * double(x) + 40), default_K, 5, wang_weights);

% Fewer scales with custom weights.
x = make_image(64, 71);
cases(end + 1) = make_case('three-scales', x, add_noise(x, 20), ...
    default_K, 3, [0.2 0.5 0.3]);

% Non-default stabilizing constants on a heavily distorted pair.
x = make_image(176, 176);
cases(end + 1) = make_case('custom-constants', x, add_noise(x, 40), ...
    [0.05 0.1], 5, wang_weights);

matlab_version = version;
here = fileparts(mfilename('fullpath'));
save(fullfile(here, 'msssim_matlab.mat'), 'cases', 'matlab_version', '-v7');
end


function c = make_case(name, x, y, K, level, weight)
win = fspecial('gaussian', 11, 1.5);
expected = msssim(x, y, K, win, level, weight, 'product');
c = struct('name', name, 'x', x, 'y', y, 'K', K, 'level', level, ...
    'weight', weight, 'expected', expected);
end


function img = make_image(rows, cols)
% Smooth structure with finer texture on top, spanning most of [0, 255].
coarse = normalize_range(imgaussfilt(randn(rows, cols), 4));
fine = normalize_range(imgaussfilt(randn(rows, cols), 0.8));
img = uint8(255 * (0.75 * coarse + 0.25 * fine));
end


function y = add_noise(x, stddev)
% uint8 conversion rounds and saturates at the ends of the range.
y = uint8(double(x) + stddev * randn(size(x)));
end


function a = normalize_range(a)
a = (a - min(a(:))) / (max(a(:)) - min(a(:)));
end
