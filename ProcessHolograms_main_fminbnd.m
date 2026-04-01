% Reconstructs holograms and locates particles using a two-stage approach:
% 1. Coarse scan: Reconstruct at large z steps to detect particle positions
% 2. Fine scan: Use optimization (fminbnd) to precisely locate z position for each particle

clear
close all

%% Input Parameters
maxNumCompThreads(4);

% Data paths
DiskName='F:\';                               % Root directory containing input folders
FoldersInput={'Holography_sample_test1'};     % Folders containing hologram files
InputHologramsFrame=[3 4];                    % Frame indices to process
FrameToStart=12;                              % Starting frame for background calculation

% Physical parameters
ReccordWavelength=449*1e-9;                  % Recording wavelength (m)
PixelsSize=3.2*1e-6;                         % Pixel size (m)
ResitutionWaveLength=640*1e-9;               % Reconstruction wavelength (m)

% Z-axis scanning range
Dz = 1e-3;                                   % Base step size (m)
Zmin = 0.25;                                 % Minimum z distance (m)
Zmax = 0.5;                                  % Maximum z distance (m)
Z_restitution = Zmin:Dz:Zmax;
Nz = numel(Z_restitution);

CPUunit=8;                                   % Number of CPU cores for parallel computation

%% Coarse and Fine Scan Parameters
coarseFactor = 5;                            % Coarse scan step multiplier
coarseDz = Dz * coarseFactor;                % Coarse scan step size
coarseZ = Zmin:coarseDz:Zmax;               % Coarse scan z values

fineHalfRange = 5e-4;                       % Fine scan range: ± this value around coarse peak (m)
Dz_fine = 1e-5;                              % Fine scan step size (m) - used for optimization tolerance

% Detection threshold
min_foreground_px = 10;                      % Minimum foreground pixels required after binarization

% Initialize parallel computing pool
p = gcp('nocreate');
if isempty(p)
    parpool('local', CPUunit);
end
%% Main Processing Loop
% Iterate over all input folders and hologram files
for indFolder=1:numel(FoldersInput)
    folderPath = fullfile(DiskName, FoldersInput{indFolder});
    listing = dir(folderPath);
    listing(ismember({listing.name},{'.','..'})) = [];
    listing(contains({listing.name},'.txt')) = [];
    [Nholo,~] = size(listing);
    
    for indHolo=1:Nholo
        % Setup output directories
        holoFile = fullfile(folderPath, listing(indHolo).name);
        FolderOutput = fullfile('D:\', listing(indHolo).name(1:end-4));
        if exist(FolderOutput,'dir')~=7, mkdir(FolderOutput); end
        Folder_saveRes = fullfile(FolderOutput,'Results');
        if exist(Folder_saveRes,'dir')~=7, mkdir(Folder_saveRes); end
        
        for indInput = 1:numel(InputHologramsFrame)
            %% Hologram Normalization
            disp(['Processing frame: ' num2str(InputHologramsFrame(indInput))]);
            tic
            HologramNormalized = Normalization(holoFile, FrameToStart, InputHologramsFrame(indInput));
            [Ny, Nx] = size(HologramNormalized);
            
            % Apply Gaussian filter and convert to uint8
            HologramNormalized = imgaussfilt(HologramNormalized,'Filtersize',3); 
            HologramNormalized = uint8(255 * HologramNormalized);
            toc
            
            %% Coarse Scan Reconstruction
            % Reconstruct hologram at coarse z steps in parallel
            disp('Starting coarse restitution...')
            tCoarse = tic;
            Zc = coarseZ;
            Nc = numel(Zc);
            
            % Preallocate memory for reconstruction stack
            try
                R1_stack = single(zeros(Ny, Nx, Nc)); 
            catch ME
                error('Memory Error. Reduce coarseFactor or image size.');
            end
            
            % Parallel reconstruction at all coarse z planes
            hologConst = parallel.pool.Constant(HologramNormalized);
            parfor k = 1:Nc
                z_k = Zc(k);
                [~, R1_k, R2_k] = Fct_restitution(hologConst.Value, z_k, ResitutionWaveLength, PixelsSize, 0);
                Rmag = abs(R1_k + 1i * R2_k);
                % Store raw intensity (not normalized) for correct minimum projection
                R1_stack(:,:,k) = single(Rmag); 
            end
            clear hologConst
            
            % Compute minimum intensity across all z planes
            RawImageMin = squeeze(min(R1_stack, [], 3));
            ImageMin = mat2gray(RawImageMin); 
            
            imwrite(ImageMin, fullfile(FolderOutput,'ImageMin_coarse.tif'),'compression','none');
            toc(tCoarse)
            
            %% Particle Detection
            % Detect particles using statistical thresholding method
            disp('Starting Detection (Statistical Method)...')
            tDetect = tic;
            
            % Step 1: Invert image (particles appear as bright peaks on dark background)
            ImDetect = imcomplement(ImageMin); 
            
            % Step 2: Background subtraction using morphological opening
            % Removes uneven illumination and large interference fringes
            Background = imopen(ImDetect, strel('disk', 15));
            ImDetect = ImDetect - Background;
            
            % Step 3: Denoise with Gaussian filter
            ImDetect = imgaussfilt(ImDetect, 2); 
            
            % Step 4: Statistical thresholding (Mean + K * Std)
            % More robust than Otsu for sparse particle detection
            detect_vals = ImDetect(:);
            mu = mean(detect_vals);
            sigma = std(detect_vals);
            K_factor = 4.0;  % Threshold factor
            T_stat = mu + K_factor * sigma;
            
            imageBin = ImDetect > T_stat;
            
            % Step 5: Remove small noise regions
            imageBin = bwareaopen(imageBin, 5);
            
            % Check detection results
            num_pixels = nnz(imageBin);
            disp(['  Detection: Found ' num2str(num_pixels) ' pixels.']);
            
            if num_pixels < min_foreground_px
                warning('  No objects detected even with statistical threshold. Skipping.');
                imwrite(ImDetect, fullfile(FolderOutput, 'Debug_DetectionMap.tif'));
                clear R1_stack
                continue; 
            end
            
            % Dilate to merge fragmented particles
            Imagebin_dil = imdilate(imageBin, strel('disk', 3));
            imwrite(Imagebin_dil, fullfile(FolderOutput,'ImageBin_coarse.tif'),'compression','none');
            
            % Extract connected components and properties
            CC = bwconncomp(Imagebin_dil);
            stats = regionprops('table', CC, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'EquivDiameter', 'SubarrayIdx');
            
            % Filter particles by size
            if ~isempty(stats)
                idx_small = find([stats.MajorAxisLength] < 3);
                stats(idx_small,:) = [];
            end
            
            % Save detection visualization
            fig = figure('Visible','off','Units','normalized','Position',[0 0 1 1]);
            imshow(ImageMin, [], 'InitialMagnification','fit')
            hold on
            if ~isempty(stats)
                centroids = cat(1, stats.Centroid);
                radii = 0.5 * mean([stats.MajorAxisLength stats.MinorAxisLength], 2);
                viscircles(centroids, radii, 'Color', 'r');
            end
            saveas(fig, fullfile(FolderOutput, sprintf('DetectedObjectAll_coarse.png')));
            close(fig);
            
            disp(['  Detection done. Objects found: ' num2str(size(stats,1))])
            toc(tDetect)
            
            %% Coarse Focus Metric Calculation
            % Compute Sobel energy (focus metric) for each particle at each coarse z plane
            disp('Computing coarse focus metric...')
            tMetric = tic;
            Nobject = size(stats, 1);
            if Nobject == 0
                warning('No objects valid after size filtering.');
                clear R1_stack
                continue;
            end
            
            % Build regions of interest (ROIs) for each particle
            ROIs = zeros(Nobject,4); 
            for ii = 1:Nobject
                cen = stats.Centroid(ii,:);
                diam = double(stats.EquivDiameter(ii));
                % Add padding to ensure fringe patterns are captured
                padding = 10; 
                x0 = round(cen(1) - diam/2 - padding);
                y0 = round(cen(2) - diam/2 - padding);
                w = round(diam + 2*padding);
                h = round(diam + 2*padding);
                
                % Clamp to image boundaries
                x0 = max(1, min(Nx, x0));
                y0 = max(1, min(Ny, y0));
                if x0 + w - 1 > Nx, w = Nx - x0 + 1; end
                if y0 + h - 1 > Ny, h = Ny - y0 + 1; end
                ROIs(ii,:) = [x0, y0, w, h];
            end
            
            % Preallocate focus metric matrix
            Sobel_coarse = zeros(Nobject, Nc);
            Bx = [-1,0,1; -2,0,2; -1,0,1]; 
            By = Bx';
            
            % Compute Sobel energy for each particle at each coarse z
            for k = 1:Nc
                slice = R1_stack(:,:,k); 
                for obj = 1:Nobject
                    r = ROIs(obj, :);
                    patch = slice(r(2):(r(2)+r(4)-1), r(1):(r(1)+r(3)-1));
                    if isempty(patch)
                        Sobel_coarse(obj, k) = 0; continue;
                    end
                    Yx = filter2(Bx, patch);
                    Yy = filter2(By, patch);
                    G = abs(sqrt(Yx.^2 + Yy.^2));
                    Sobel_coarse(obj, k) = sum(G(:));
                end
            end
            toc(tMetric)
            
            %% Determine Fine Scan Ranges
            % Find peaks in coarse focus metric to determine fine scan z ranges
            disp('Selecting coarse peaks...')
            z_candidates_per_object = cell(Nobject,1);
            
            for obj = 1:Nobject
                metric = Sobel_coarse(obj, :);
                % Smooth the metric
                metric_s = movmean(metric, 3);
                
                % Find peaks
                [pks, locs] = findpeaks(metric_s);
                
                % Fallback if no peaks found
                if isempty(pks)
                    [~, idxMax] = max(metric_s);
                    locs = idxMax; pks = metric_s(idxMax);
                else
                    % Filter out low peaks
                    th = min(metric_s) + 0.3 * (max(metric_s) - min(metric_s));
                    keep = pks > th;
                    locs = locs(keep);
                end
                
                % Convert peak indices to z values
                zvals = Zc(locs);
                z_candidates_per_object{obj} = zvals;
            end
            
            %% Fine Scan Using Optimization
            % Use fminbnd optimization to precisely locate z position for each particle
            disp('Starting fine scan (Optimization Method)...')
            tFine = tic;
            HologConst = parallel.pool.Constant(HologramNormalized);
            z_final = nan(Nobject,1);
            
            % Parallel optimization for each particle
            parfor obj = 1:Nobject
                z_cands = z_candidates_per_object{obj};
                if isempty(z_cands), continue; end

                r = ROIs(obj, :); % ROI coordinates [x, y, w, h]
                
                % Use first coarse peak as optimization center
                z_center = z_cands(1); 
                z_lower = max(Zmin, z_center - fineHalfRange);
                z_upper = min(Zmax, z_center + fineHalfRange);
                
                % Define cost function: fminbnd minimizes, so negate focus metric
                CostFunc = @(z) -1 * ComputeFocusMetric(HologConst.Value, z, r, ResitutionWaveLength, PixelsSize);
                
                % Optimization options: high precision (TolX 1e-6 ≈ 1 um)
                options = optimset('TolX', 1e-6, 'Display', 'off');
                
                try
                    z_best = fminbnd(CostFunc, z_lower, z_upper, options);
                    z_final(obj) = z_best;
                catch
                    z_final(obj) = NaN;
                end
            end
            toc(tFine)
            
            %% Results Saving and Visualization
            % Extract and save particle positions
            Result = zeros(Nobject, 3); % [X, Y, Z]
            for obj = 1:Nobject
                cen = stats.Centroid(obj,:);
                Result(obj,:) = [cen(1), cen(2), z_final(obj)];
            end
             
            % Save results to MAT file
            save(fullfile(FolderOutput, sprintf('Results_coarse_fine.mat')), 'Result', 'ImageMin', 'z_final');
            
            % Create scatter plot visualization
            fig = figure('Visible','off');
            imshow(ImageMin,'InitialMagnification','fit'); hold on;
            scatter(Result(:,1), Result(:,2), 20, 'r', 'filled');
            title('Detected Objects (Red)');
            saveas(fig, fullfile(FolderOutput, sprintf('Scatter_coarse_fine.png')));
            close(fig);
            
            % Cleanup large variables
            clear R1_stack Sobel_coarse
            disp('Done.')
            
        end 
    end 
end

%% Helper Function: Focus Metric for Optimization
% Computes Sobel energy (focus metric) at a specific z position
% Used by fminbnd to find optimal z position
function val = ComputeFocusMetric(Hologram, z, roi, lambda, px_size)
    % Step 1: Reconstruct hologram at specified z
    [~, R1, R2] = Fct_restitution(Hologram, z, lambda, px_size, 0);
    
    % Step 2: Crop to ROI region
    r_idx = roi(2):(roi(2)+roi(4)-1);
    c_idx = roi(1):(roi(1)+roi(3)-1);
    
    % Boundary safety check
    [Ny, Nx] = size(R1);
    r_idx = r_idx(r_idx>0 & r_idx<=Ny);
    c_idx = c_idx(c_idx>0 & c_idx<=Nx);
    
    patch_real = double(R1(r_idx, c_idx));
    patch_imag = double(R2(r_idx, c_idx));
    
    % Step 3: Calculate Sobel gradient energy
    Bx = [-1,0,1; -2,0,2; -1,0,1]; 
    By = Bx';
    
    % Complex gradient computation
    Gx = conv2(patch_real, Bx, 'valid') + 1i * conv2(patch_imag, Bx, 'valid');
    Gy = conv2(patch_real, By, 'valid') + 1i * conv2(patch_imag, By, 'valid');
    
    % Return total gradient energy as focus metric
    val = sum(sum(abs(Gx) + abs(Gy)));
end