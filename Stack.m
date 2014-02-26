classdef Stack < handle
   
   properties
      stack
      stack2
      width, height
      dt, T, frames
      lsmMeta
      stim, stimDt
      nnIdx, nnDst, spatialDist
   end
   
   methods (Access='public')
      
      function obj = Stack(varargin)
         % constructor for the STACK object
         % USAGE
         % s = Stack(X), where X is a 3D matrix (width x height x time)
         % s = Stack(X, dt), dt is 1/framerate
         % s = Stack(pathToLsmFile) (relies on lsminfo and tiffread30b)
         if isnumeric(varargin{1}) | islogical(varargin{1})
            obj.stack  = varargin{1};
            obj.width  = size(obj.stack,1);
            obj.height = size(obj.stack,2);
            obj.frames = size(obj.stack,3);
            if nargin>1
               obj.dt = varargin{2};
            else
               obj.dt = 1;
            end
            obj.T = 1:obj.dt:obj.frames;
         else
            obj.readLSM(varargin{1});
         end
         obj.stack = single(obj.stack);
      end
      
      function readLSM(obj,lsmFilePath)
         % load stack from LSM file. called by the constructor.
         % TODO:
         %  make private? NO
         %  reduce dependency on external functions (lsminfo, tiffread30b)
         tiff = tiffread30b(lsmFilePath);
         try % try to read meta data from LSM file
            obj.lsmMeta = lsminfo(lsmFilePath);
            obj.frames = length(obj.lsmMeta.TimeStamps.TimeStamps');
            obj.T = obj.lsmMeta.TimeStamps.TimeStamps';
         catch
            obj.frames = length(tiff);
            obj.T = 1:obj.frames;
         end
         obj.dt = mean(diff(obj.T));
         if iscell(tiff(1).data)
            firstFrame = tiff(1).data{1};
         else
            firstFrame = tiff(1).data;
         end
         dataType =  class(firstFrame);
         if any(size(firstFrame)==1)
            obj.width = tiff(1).width;
            obj.height = tiff(1).height;
         else
            [obj.width, obj.height] = size(firstFrame);
         end
         
         obj.stack = zeros(obj.width, obj.height, obj.frames, dataType);
         if length(tiff(1).data)>1
            obj.stack2 = zeros(obj.width, obj.height, obj.frames, dataType);
         end
         for t = obj.frames:-1:1
            if iscell(tiff(t).data)
               thisFrame = reshape(tiff(t).data{1}, obj.width, obj.height);
               if length(tiff(t).data)>1
                  thisFrame2 = reshape(tiff(t).data{2}, obj.width, obj.height);                  
                  obj.stack2(:,:,t) = thisFrame2;
               end
            else
               thisFrame = reshape(tiff(t).data   , obj.width, obj.height);
            end
            obj.stack(:,:,t) = thisFrame;
         end
      end
      
      function setStim(obj,stim, dt)
         obj.stim = stim;
         obj.stimDt = dt;
      end
      
      function flatten(obj)
         % converts each 2D-frame into a 1D vector. convenient for vector
         % operations
         obj.stack = reshape(obj.stack, obj.width*obj.height, obj.frames);
      end
      
      function unflatten(obj)
         % converts the 1D vector representation back into 2D images
         obj.stack = reshape(obj.stack, obj.width, obj.height, obj.frames);
      end
      
      function align(obj, varargin)
         % align using the matlab imregistration function (R2012a+).
         % s.aling([IMG],[MODE])
         % ARGS
         %  IMG  - OPTIONAL, use IMG as a template; if omitted, the aligns
         %                   the temporal average frame is used
         %  MODE - OPTIONAL, specify class of transformations implemented
         %                   by imregister ('translation' - DEFAULT,
         %                   'rigid','affine'); see doc imregister for
         %                   details.
         if nargin==1 || isempty(varargin{1})
            template = mean(obj.stack,3);% temporal average as a template
         elseif ~isempty(varargin{1})
            template = varargin{1};
         end
         MODE = 'translation';
         if nargin==3
            MODE = varargin{2};
         end
         [optimizer,metric] = imregconfig('monomodal');
%          % get initial transform for first frame
%          [~, ~, tForm]  = imregister2(obj.stack(:,:,1), template, ...
%             MODE,optimizer,metric);
          obj.stack = mapFun(obj.stack, @imregister, {template, MODE,optimizer,metric});
%          for t = 1:size(obj.stack,3)
%             % use previous frame's transform, TFORM, as initial condition
%             % ONLY SUPPORTED R2013a?
%             [obj.stack(:,:,t), ~, tForm] = imregister2(obj.stack(:,:,t), template, ...
%                MODE,optimizer,metric, 'InitialTransformation', tForm);
%         end
      end
      
      function resize(obj, scaleFactor)
         % resize each frame
         % s.resize(SCALE)
         % ARGS
         %  SCALE - ..., uses imresize on each frame with a 'box'
         %          interpolation kernel
         obj.unflatten;
         obj.stack = mapFun(obj.stack, @imresize,{scaleFactor, 'box'});
         %for t = obj.frames:-1:1
         %   tmpStack(:,:,t) = imresize(obj.stack(:,:,t), scaleFactor, 'box');
         %end
         %obj.stack = tmpStack;
         % update frame dimensions
         [obj.width, obj.height, obj.frames] = size(obj.stack);
      end
      
      function filterSpatial(obj, flt)
         % smooth each frame using a kernel
         % s.filterSpatial(KERNEL)
         % ARGS
         %  KERNEL - ...
         obj.unflatten();
         obj.stack = mapFun(obj.stack, @imfilter, {flt, 'replicate'});
         %for t = 1:obj.frames
         %   obj.stack(:,:,t) = imfilter(obj.stack(:,:,t), flt);
         %end
      end
      
      function filterTemporal(obj, flt)
         % temporally filter each pixel using 1D filter
         % s.filterTemporal(KERNEL)
         % ARGS
         %  KERNEL - 1D vector
         % TODO: check for boundary conditions and temporal offsets
         obj.flatten;
         obj.stack = convMatrix(obj.stack,flt,'full');
         obj.stack = obj.stack(:,ceil(length(flt)/2):end-floor(length(flt)/2));
      end
      
      function ic = ica(obj)
         % perform temporal(?) ICA. needs FASTICA
         % ic = s.ica()
         % RETURNS
         %  IC - struct with fields
         %     feat   - temporal traces
         %     sepMat - basis for the temporal IC traces
         %     mixMat - basis for getting original traces from feat
         ic.n = 12;
         obj.flatten();
         [ic.feat, ic.sepMat, ic.mixMat] = fastica(obj.stack', 'verbose', 'off', 'displayMode', 'off','lastEig',ic.n);
      end
      
      function sf = sfa(obj)
         % perform temporal SLOW-FEATURE ANALYSIS
         obj.flatten()
         sf.n = 12;
         sf.ppDim = 127;
         sf.ppType = 'SFA1';
         sf.derivType = 'ORD3a';
         % create an SFA object and get a reference to it
         hdl = sfa2_create(sf.ppDim, sf.n, sf.ppType, sf.derivType);
         chunkIdx = linspace(1,obj.width*obj.height,10);
         % cycle over the two SFA steps and chunks
         for step_name = {'preprocessing', 'expansion'},
            for i = 1:length(chunkIdx)-1,
               % cut part of data set
               x = obj.stack(chunkIdx(i):chunkIdx(i+1),:)';
               % update the current step
               sfa_step(hdl, x, step_name{1});
            end
         end
         % close the algorithm
         sfa_step(hdl, [], 'sfa');
         sf.feat = sfa_execute(hdl,obj.stack);
         sfa_clearall()
         sf.trace = obj.stack*sf.feat;
         obj.unflatten();
      end
      
      function normalize(obj)
         % normalize as in the AHRENS Science paper
         % TODO: extend to allow for standard df/f normalization
         obj.flatten;
         tempMean = mean(obj.stack,2);
         obj.stack = bsxfun(@minus,obj.stack, tempMean);
         obj.stack = bsxfun(@times,obj.stack, 1./(tempMean-mean(obj.stack(:))));
      end
      
      
      function play(obj)
         % play stack as a sequence of frames
         obj.unflatten();
         maxpixel=max(obj.stack(:));
         minpixel=min(obj.stack(:));
         for t=1:obj.frames
            figure(100);
            myPcolor(obj.stack(:,:,t));
            caxis([minpixel maxpixel]);
            drawnow;
         end
      end
      
      function plot(obj,frame)
         % display individual frame
         imagesc(obj.stack(:,:,frame));
      end
      
      function m = getCorrMap(obj)
         % as in AHRENS Science paper. determines temporal corr of each pix with its
         % 8-pix neighborhood. useful for detecting locally correlated
         % pixels
         obj.unflatten;
         m = zeros(obj.width, obj.height);
         for x = 2:obj.width-1
            for y = 2:obj.height-1
               tmp = reshape(obj.stack(x+(-1:1),y+(-1:1),:),9,[]);% get pixels in a one-pix neighbour surrrounding x,y
               m(x,y) = mean(pdist2(tmp(5,:),tmp([1:4 6:9],:),'correlation'));% avg. corr between center and surround pix
            end
         end
         m = (1-m).^2;
         m = im2bw(m,mean(m(:)) + 0*std(m(:)));
      end
      
      function m = getCubicMap(obj)
         obj.flatten
         m = abs(mean(obj.stack.^3,2));
         m = im2bw(m,mean(m(:)) + 0*std(m(:)));
      end
      
      function [roiTrace, roiTraces] = roiTrace(obj,roiMask)
         % [roiTrace, roiTraces] = roiTrace(obj,roiMask);
         % ARGS
         %  ROIMASK - binary matrix, width x height x (number of rois) - if
         %  left empty will assume the full frame is the ROI
         % RETURNS
         %  roiTrace - average trace for each roi (time x (number of rois))
         % roiTraces - traces of each pixel in the roi (cell array of size
         %             (number of rois)
         if nargin==1, roiMask = ones(obj.width, obj.height,1, 'int8'); end
         obj.flatten();
         roiStack = Stack(roiMask);
         roiStack.flatten();
         for r = 1:roiStack.frames
            roiTraces{r} = bsxfun(@times, obj.stack(roiStack.stack(:,r)>0,:), roiStack.stack(roiStack.stack(:,r)>0,r));
            roiTrace(:,r)  = mean(roiTraces{r});
         end
      end
      
      function [newROI] = roiClean(obj, roi, thres)
         % Clean ROI. Remove pixels from ROI that are less than THRES
         % correlated with ROI trace
         % [ROInew, ROInewTrace] = s.roiClean(ROIold, THRES)
         % ARGS
         %  ROIold - rois (width x height x n rois)
         %  THRES - minimal correlation of individual pix to roi traces
         % RETURNS
         %  ROInew - new and cleaner ROIs  (width x height x n rois)
         
         obj.flatten;
         roiStack = Stack(roi);
         roiStack.flatten();
         newROI = zeros(roiStack.width, roiStack.height, roiStack.frames);
         
         for r = 1:roiStack.frames
            roiIdx = find(roiStack.stack(:,r)>0);
            for run = 1:8
               %C = nancov(obj.stack(roiIdx,:));
               %D = pdist(obj.stack(roiIdx,:),'correlation');
               %Z = linkage(D);
               %C = cluster(Z,'maxclust',10);
               %[cnt, bin] = hist(C,1:max(C));
               %[cntVal, cntIdx] = sort(cnt,'descend');
               %C = map(C,[ cntIdx;1:max(C)]);
               %[~, sortIdx] = sort(C);
               D2 = 1-pdist2(mean(obj.stack(roiIdx,:),1), obj.stack(roiIdx,:),'correlation');
               D2 = sign(D2).*(D2.^2);
               if ~any(D2<thres) || sum(D2>thres)<2
                  break
               end
               roiIdx(D2<thres) = [];
            end
            % built new roi image
            roi = zeros(roiStack.width, roiStack.height);
            roi(roiIdx)  = D2;
            newROI(:,:,r) = reshape(roi, roiStack.width, roiStack.height);
         end
      end
      
      function newROI = roiGrow(obj, rois, thres)
         % grow ROI to include nearby pixel that are correlated with the
         % ROIs' trace
         % ROInew = s.roiGrow(ROIold, THRES)
         % ARGS
         %  ROIold - binary mask
         %  THRES  - minimal correlation necessary for inclusion
         % RETURNS
         %  ROInew - new binary mask comprising the old ROI and pixels
         %           whose temporal correlation with the ROI trace is >THRES
         
         roiStack = Stack(rois);
         % get templates
         templateTraces = obj.roiTrace(rois);
         for r = 1:roiStack.frames
            roi = roiStack.stack(:,:,r);
            oldRoi = roi;
            % grow roi using bwdist
            roi = bwdist(roi)<20 & oldRoi==0;
            % get new traces
            [~, roiTraces] = obj.roiTrace(roi);
            % get sim of all traces to template
            D2 = 1-pdist2(templateTraces(:,r)', roiTraces{1},'correlation');
            D2 = sign(D2).*(D2.^2);
            % reject all pixels with r2<thres
            roiIdx = find(roi);
            roi(roiIdx(D2<thres)) = 0;
            roi(oldRoi>0) = 1;
            newROI(:,:,r) = roi;
         end
         %          %%
         %                   % get template
         %          templateTrace = obj.roiTrace(roi);
         %          % grow roi using bwdist
         %          oldRoi = roi;
         %          roi = bwdist(roi)<20 & oldRoi==0;
         %          % get new traces
         %          [~, roiTraces] = obj.roiTrace(roi);
         %          % get sim of all traces to template
         %          D2 = 1-pdist2(templateTrace, roiTraces,'correlation');
         %          D2 = sign(D2).*(D2.^2);
         %          % reject all pixels with r2<thres
         %          roiIdx = find(roi);
         %          roi(roiIdx(D2<thres)) = 0;
         %          roi(oldRoi>0) = 1;
      end
      
      function [allRois, allCCN, nRois, thres, sizes] = roiFindKNNGridSearch(obj, varargin)
         % perform grid search for best combination of similarity and size
         % [allRois, allCCN, nRois, thres, sizes] = roiFindKNNGridSearch(obj)
         if nargin>2
            mask = varargin{1};
         else
            mask = true(obj.width, obj.height);
         end
         
         thres = .05:.05:1;
         sizes = 2:2:36;
         for si = 1:length(sizes)
            for idx = 1:length(thres)
               fprintf('.')
               [allRois{idx,si}, allCCN(:,:,idx,si)] = obj.roiFindKNN(thres(idx), sizes(si), mask);
               thisCCN = allCCN(:,:,idx,si);
               nRois(si, idx) = size(allRois{idx,si}, 3);
            end
         end
         
      end
      
      
      function [rois, CCN, nnMap] = roiFindKNN(obj,thres, minSize,varargin)
         % find ROIs based on nearest-neighbour networks
         % [rois, CCN] = roiFindKNN(thres, minSize, [mask])
         % ARGS
         %  thres    - minimal similarity between pixel traces to be considered neighbours between >0 and <1
         %  minSize  - minimal size (in pixel) of an ROI
         %  mask     - OPTIONAL, binary mask useful to ignore noise pixels and speed up algorithm
         % RETURNS
         %  rois  - stack of binary ROI masks (width x height x number of ROIs)
         %  CCN   - image color coding ROIs
         %  nnMap - proximity network of pixels
         
         if nargin>3
            mask = varargin{1};
         else
            mask = true(obj.width, obj.height);
         end
         obj.flatten();
         if isempty(obj.nnIdx)
            [obj.nnIdx,obj.nnDst] = knnsearch(obj.stack(mask,:), obj.stack(mask,:), 'k', 10, 'Distance', 'correlation');
            obj.nnIdx = obj.nnIdx(:,2:end);
            obj.nnDst = obj.nnDst(:,2:end);
            %          D = squareform(pdist(obj.stack, 'correlation'));
            %          [sortDst, sortIdx] = sort(D);
            %          k = 10;
            %          obj.nnDst = sortDst(2:k+1,:)';
            %          obj.nnIdx = sortIdx(2:k+1,:)';
         end
         % get spatial dist between pix
         %%
         nnIdx = obj.nnIdx;
         maskIdx = find(mask);
         [nnIdxX,nnIdxY] = ind2sub([obj.width, obj.height],maskIdx);
         nnIdxX = nnIdxX(nnIdx);
         nnIdxY = nnIdxY(nnIdx);
         [idxX,idxY] = ind2sub([obj.width, obj.height],1:obj.width*obj.height);
         idxX = idxX(maskIdx);
         idxY = idxY(maskIdx);
         dx = bsxfun(@minus, nnIdxX, idxX');
         dy = bsxfun(@minus, nnIdxY, idxY');
         spatialThres = 160;
         obj.spatialDist = sqrt((dx.^2 + dy.^2));%/sqrt(obj.width.^2 + obj.height.^2);
         % remove pix too distant
         %nnIdx(obj.spatialDist(:)>spatialThres) = nan;
         nnDst = 1-(1-obj.nnDst).*(exp(-obj.spatialDist/(2*spatialThres)));
         % remove pix with too disimilar traces - weight by spatial distance?
         %nnIdx((1-obj.spatialDist(:)).*obj.nnDst(:)>thres) = nan;
         nnIdx(nnDst(:)>thres) = nan;
         % find idx that are neighbours to each other - threshold by minimal spatial distance?
         %nnMap = zeros(sum(mask(:)>0), 10);
         nnMap = zeros(sum(mask(:)>0));
         %nnMap = zeros(nanmax(nnIdx(:)));
         for idx = 1:size(nnMap,1)
            nni = nnIdx(idx,:);
            nni(isnan(nni)) = [];
            nnMap(idx,nni) = 1;
         end
         %%
         % get a bunch of non-overlapping ROIs
         [~, C] = graphconncomp(sparse(nnMap),'Directed','true','weak','true');
         %[C, L, U] = SpectralClustering(nnMap, 10, 3);
         
         % remove too small clusters
         ccSize = histc(C,unique(C));% count cluster size
         C(ismember(C,find(ccSize<minSize))) = 0;% mark all small clusters
         [Cn] =  grp2idx(C);% give multi-node clusters consecutive labels
         % transform cluster back from mask coord's to frame coord's
         CCN = nan(size(mask));
         CCN(mask) = Cn;
         CCN(CCN(:)==1) = nan;% set clusters marked as 0 to nan
         uniG = unique(CCN);
         uniG(isnan(uniG)) = [];
         rois = zeros(obj.width, obj.height, length(uniG));
         for idx = 1:length(uniG)
            rois(:,:,idx) = CCN==uniG(idx);
         end
      end
      
      function roiPlot(obj, ROIs)
         % roiPlot(ROIs)
         [trace, traces] = obj.roiTrace(ROIs);
         roiPlot = nan(obj.width, obj.height);
         cmap = colormap(jet(size(ROIs,3)));
         cmap = bsxfun(@plus, cmap, -mean(cmap) + 0.5);
         cmap(cmap>1) = 1;
         cmap(cmap<0) = 0;
         colormap(cmap);
         subplot(122)
         normTrace = normalize(trace')';
         offSet = mean(range(normTrace));
         hold on
         for idx = length(traces):-1:1
            traceCorr(idx) = 1-mean(pdist(normalizeMean(traces{idx}),'correlation'));
            roiPlot(ROIs(:,:,idx)>0) = idx;
            co = cmap(idx,:);
            co = co+.5; co(co>1) = 1;
            plot(normalize(traces{idx})'+idx*offSet,'Color',co,         'LineWidth',0.5);
            plot(normTrace(:,idx)       +idx*offSet,'Color',cmap(idx,:),'LineWidth',1.5);
            drawnow
         end
         hold off
         %axis('off','tight')
         
         subplot(221)
         myPcolor(roiPlot)
         axis('square')
         subplot(223)
         hl = plot(trace);
         for idx=1:length(hl),set(hl(idx),'Color',cmap(idx,:),'LineWidth',1.5),end
         axis('tight','off')
      end
      
      function drawROItrace(obj)
         disp('Not implemented yet!')
      end
      
      
   end
end

