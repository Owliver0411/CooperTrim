"""
==============================================================================
FILE RECAP: point_pillar_cobevt.py   (3D_Detection_OPV2V version)
Original location: 3D_Detection_OPV2V/opencood/models/
==============================================================================

WHAT THIS FILE IS
-----------------
The complete neural network ("model") used for 3D object detection from LiDAR
on the OPV2V dataset, with CooperTrim built in. It contains one class,
PointPillarCoBEVT. The config line `core_method: point_pillar_cobevt` (in the
`model:` section of the YAML) selects it, and tools/train.py trains it.

Input : the LiDAR scans of several cooperating cars in one traffic scene.
Output: for every location on a bird's-eye-view grid, (a) a score for "is there
        a vehicle here?" and (b) numbers describing its box (position, size,
        heading).

THREE KINDS OF CODE ARE MIXED IN THIS FILE
------------------------------------------
 1. PointPillars  (existing, not CooperTrim): turns each car's LiDAR points
    into a feature map.
 2. CoBEVT fusion (existing, not CooperTrim): merges the feature maps of
    several cars into one.
 3. CooperTrim    (the paper's contribution; the original code marks these
    pieces with "#CooperTrim ..." comments): decides which channels of the
    feature map the ego car asks its neighbours for, and measures the
    uncertainty that drives that decision.

DATA FLOW OF forward()
----------------------
 LiDAR points of all cars
   -> PillarVFE          : points grouped into "pillars"; one vector per pillar
   -> PointPillarScatter : pillar vectors placed on a 2D grid (a pseudo-image)
   -> BaseBEVBackbone    : a CNN extracts richer features from that grid
   -> shrink_conv        : reduces 384 channels to 256
   -> [CooperTrim] divide_batches -> create_request -> create_response
                   -> accumulate_features_at_ego
   -> (optional) naive_compressor
   -> regroup            : arranges features as [scene, car slot, C, H, W]
   -> fusion_net         : CoBEVT fusion merges the cars' features
   -> cls_head, reg_head : the two output maps
 (The ego car's fused map is also saved, to be used as "the previous frame"
  by CooperTrim on the next call.)

SHAPES FOR THE OPV2V CONFIG (configs/3D_Detection_OPV2V/*.yaml)
---------------------------------------------------------------
 * Scanned area: x from -140.8 to 140.8 m, y from -38.4 to 38.4 m.
 * Pillar size 0.4 m  ->  a 704 x 192 grid of pillars.
 * The backbone's first stage has stride 2  ->  feature map of 352 x 96,
   so H = 96 and W = 352.
 * 3 x 128 = 384 channels out of the backbone, reduced to 256 by shrink_conv.
 * So each car's feature map is [256, 96, 352] (about 8.6 million numbers).
 * Up to max_cav = 5 cars cooperate in one scene.

GLOSSARY
--------
 LiDAR      : a laser scanner; its output is a "point cloud" (x, y, z points).
 BEV        : bird's-eye view, i.e. a top-down 2D map of the scene.
 Feature map: a stack of 2D grids of numbers produced by a neural network. Each
              grid is a "channel" (one kind of pattern). Shape [C, H, W].
 CAV        : connected autonomous vehicle (one of the cooperating cars).
 Ego        : the car that is receiving help; by OpenCOOD's convention it is
              index 0 among the cars of a scene.
 record_len : list saying how many cars each scene in the batch contains. All
              cars of all scenes are stacked in one tensor; this tells the code
              where to cut.
 Anchor     : a template box placed at every grid cell (2 per cell here); the
              model predicts a score and a correction for each one.
 psm / rm   : "probability score map" and "regression map", the two outputs.
 Epoch      : one full pass over the training data.

THINGS WORTH KNOWING (my reading of the code; I have not run it)
----------------------------------------------------------------
 1. The cooperation is SIMULATED. All cars' features are computed inside this
    one program; "request" and "response" just pick channels by index. The
    bandwidth is measured as the percentage of channels selected; nothing is
    actually sent over a network.
 2. "Previous frame" means whatever the previous call of forward() produced
    (self.prev_fused_feature). In the single-GPU path of tools/train.py the
    training loader uses shuffle=True, so during training that is a random
    other scene, not the previous moment in time. The validation loader uses
    shuffle=False (consecutive frames, if the dataset is stored in time order;
    I did not check that).
 3. The channel selector's layers, its threshold, and `confidence_level` look
    like they receive no gradient: the mask is a hard 0/1 comparison, the chosen
    channels are turned into integer indices, the selected percentage is a
    plain Python number, and `.item()` is used on confidence_level. See the
    notes at those lines.
 4. For channels that were NOT requested, every car's slot is filled with a
    copy of the ego car's own channel, so the fusion network never sees empty
    data.
 5. Sizes (256 channels, 96 x 352) are hard-coded for the selector.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

# Building blocks that come from the existing OpenCOOD framework (not written for
# CooperTrim). Each one is explained where it is used in __init__ below.
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup
#CooperTrim autonet
# numpy: used for np.percentile in compute_conformal_uncertainty.
import numpy as np
# The CooperTrim selector network (see channel_select_attention.py): decides which
# channels to request.
from opencood.models.sub_modules.channel_select_attention import CrossAttentionMaskPredictorAdaptive
# Not used anywhere in this file (also `rearrange` above is unused).
import os
# random: used for the epsilon-greedy coin flip in create_request.
import random


# The whole detection model. As an nn.Module, PyTorch tracks every layer created
# in __init__ (to train, save and load them), and calling model(...) runs forward().
class PointPillarCoBEVT(nn.Module):
    # __init__ builds all the layers once. `args` is the `model: args:` dictionary
    # from the YAML config file.
    def __init__(self, args):
        super(PointPillarCoBEVT, self).__init__()

        # Maximum number of cars that cooperate in one scene (5 in the config).
        self.max_cav = args['max_cav']
        # ---- PART 1: PointPillars encoder (existing code) -----------------------------
        # PillarVFE ("voxel feature encoder"): LiDAR points are bucketed into "pillars"
        # (columns of space, 0.4 m x 0.4 m, full height). A small network turns the points
        # inside each pillar into one vector of 64 numbers.
        # num_point_features=4 means each LiDAR point has 4 values (x, y, z, intensity).
        # voxel_size and point_cloud_range (the scanned area) come from the config.
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        # PointPillarScatter: puts each pillar's vector at its position on a 2D grid,
        # producing an image-like tensor (64 channels) that ordinary 2D convolutions can use.
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # BaseBEVBackbone: a 2D convolutional network (CNN). It works at three scales
        # (64, 128 and 256 filters), upsamples the results back to the same size and
        # concatenates them: 3 x 128 = 384 channels at half the grid resolution.
        # The "64" is the number of input channels it receives from the scatter step.
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # Optional "shrink" convolution (present when `shrink_header` is in the config):
        # reduces 384 channels to 256, which saves memory in the later steps. This is why
        # the feature map handed to the rest of the model has 256 channels.
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        # Optional extra channel compression (config key `compression`; 0 = disabled, as in
        # the CooperTrim detection config). It is a different bandwidth-saving technique
        # from CooperTrim's selection.
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        # ---- PART 2: CoBEVT fusion (existing code) ------------------------------------
        # SwapFusionEncoder (config section `fax_fusion`): a transformer-based block (it
        # uses attention) that merges the feature maps of all the cars into one map that is
        # centred on the ego car. This is the "fusion" that CooperTrim is applied to.
        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])

        # Output head 1 ("classification"): for each grid cell and each anchor, one score:
        # "is there a vehicle here?". A Conv2d with kernel_size=1 is a small linear map
        # applied independently at every grid cell. 128 * 2 = 256 is the number of
        # channels of the fused feature map.
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        # Output head 2 ("regression"): 7 numbers per anchor that describe how to correct
        # the template box (position x, y, z, size, and heading angle).
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        # Optionally freeze the earlier layers (used when fine-tuning). The config sets
        # backbone_fix to false, so everything is trained.
        if args['backbone_fix']:
            self.backbone_fix()

        #CooperTrim autonet
        # ---- PART 3: CooperTrim additions ---------------------------------------------
        # On/off switch. True: use CooperTrim's channel selection. False: always request
        # every channel (the baseline behaviour). It defaults to True and the CooperTrim
        # detection config does not set it.
        self.use_coopertrim = args.get('use_coopertrim', True)
        #CooperTrim channel select adapt
        # The selector network is built for a fixed feature-map size: 256 channels and a
        # 96 x 352 grid (see the header). These must match the real feature map or the
        # selector fails with a shape error. The commented-out 48 / 176 look like values
        # from a different setup.
        num_channel_select = 256
        num_spatial_select_h = 96 #48 #config['channel_select']['spatial_dim']
        num_spatial_select_w = 352 #176
        # The selector (explained in channel_select_attention.py): given the uncertainty of
        # each channel, it outputs a 0/1 mask saying which channels the ego car requests.
        self.channel_select_model = CrossAttentionMaskPredictorAdaptive(num_channels=num_channel_select, spatial_dim_h=num_spatial_select_h, spatial_dim_w=num_spatial_select_w)

        #CooperTrim autonet
        # Memory of "the previous frame": filled at the end of forward() with the ego
        # car's fused feature map, and used by the next call as the reference for
        # measuring how much the scene changed. None until the first call.
        self.prev_fused_feature = None
        # CooperTrim learnable confidence level
        # The paper's learnable quantile threshold q (Section 3.2), stored as a trainable
        # parameter that starts at 50, i.e. the 50th percentile (the median).
        # NOTE: in compute_conformal_uncertainty it is converted with `.item()` into a plain
        # Python number, which cuts the link to PyTorch's gradient tracking, so as far as I
        # can tell it never gets trained.
        initial_confidence_level = 50  # Initial value for the confidence level
        self.confidence_level = nn.Parameter(torch.tensor(initial_confidence_level, dtype=torch.float32))


    # Freezes layers: `requires_grad = False` tells PyTorch not to update that layer's
    # weights during training.
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    #CooperTrim autonet
    # Splits the stacked tensor of all cars' feature maps back into one chunk per scene.
    # Example: record_len = [3, 2] means scene 1 has 3 cars and scene 2 has 2, so a
    # tensor [5, C, H, W] becomes two tensors [3, C, H, W] and [2, C, H, W].
    def divide_batches(self, spatial_features_2d, record_len):
        """
        Divide spatial_features_2d into batches based on record_len.

        Args:
            spatial_features_2d (torch.Tensor): Tensor of size (n, c, h, w).
            record_len (list): List of integers with batch sizes. Sum of record_len = n.

        Returns:
            list: List of divided batches, where each batch corresponds to the data for a single record_len.
        """
        # The result: a list with one tensor per scene.
        divided_batches = []
        start_idx = 0

        # For each scene, cut out its cars (start_idx tracks where the previous scene ended).
        for batch_size in record_len:
            # Extract the batch slice based on record_len
            batch_data = spatial_features_2d[start_idx:start_idx + batch_size]#.to(spatial_features_2d.device)
            divided_batches.append(batch_data)
            start_idx += batch_size

        return divided_batches

    #CooperTrim autonet
    #CooperTrim conformal prediction
    # CooperTrim's uncertainty measure (paper Section 3.2 and Figure 13). It answers,
    # for each channel: "how much did this channel change compared with what the ego
    # car understood in the previous frame?" A big change means the ego car is
    # unsure about that channel, so it is a good one to request from neighbours.
    # (The word "conformal" is borrowed from conformal prediction, a statistics
    # technique; here it is only loosely inspired by it, as the paper says.)
    #
    # NOTE: the docstring below mentions shapes (128, 32, 32) from the segmentation
    # project. Here the shapes are (256, 96, 352).
    def compute_conformal_uncertainty(self, reference_data, current_data, confidence_level=90):
        """
        Compute uncertainty using conformal prediction based on reference tensor.

        Parameters:
            reference_data (torch.Tensor): Reference tensor of shape (128, 32, 32).
            current_data (torch.Tensor): Current tensor of shape (128, 32, 32).
            confidence_level (int): Confidence level for quantile threshold (e.g., 90 for 90%).

        Returns:
            quantile_threshold (float): The quantile threshold for conformity scores.
            uncertainty_flags (torch.Tensor): Boolean tensor of shape (128) indicating uncertainty.
            uncertainty_intervals (list): List of tuples containing lower and upper bounds for uncertainty intervals.
        """
        # Step 1: Compute conformity scores (mean absolute difference across spatial dimensions)
        # |previous fused feature - current feature|, cell by cell: shape [C, H, W].
        # This is the paper's "non-conformity score". `reference_data` is the previous
        # call's fused feature; `current_data` is the ego car's fresh feature map.
        conformity_scores_whole = torch.abs(reference_data - current_data)
        # Average over the H x W grid: one number per channel, shape [C]. A large value
        # means "this channel changed a lot since the last frame".
        conformity_scores = conformity_scores_whole.mean(dim=(1, 2))  # Shape: [128]

        # Step 2: Determine the quantile threshold (e.g., 90th percentile for 90% confidence)
        # Move the 256 scores to the CPU as a numpy array. `.detach()` cuts them off from
        # PyTorch's gradient tracking (numpy cannot track gradients).
        conformity_scores_np = conformity_scores.detach().cpu().numpy()

        # CooperTrim learnable confidence level
        # quantile_threshold = np.percentile(conformity_scores_np, confidence_level)  # Convert to numpy for quantile calculation
        # The cut-off: the `confidence_level`-th percentile of the 256 scores. At 50 this is
        # the median, so about half the channels score above it. A higher value means a
        # stricter cut-off and fewer flagged channels.
        # `.item()` turns the trainable parameter into a plain Python number, which is
        # where any link to gradient tracking is lost (see the note in __init__).
        quantile_threshold = np.percentile(conformity_scores_np, confidence_level.item())

        # Step 3: Inductive uncertainty quantification
        # Flag channels where the conformity score exceeds the quantile threshold
        # True for channels whose score is above the cut-off ("uncertain" channels). Shape [C].
        uncertainty_flags = conformity_scores > quantile_threshold  # Shape: [128]

        # Step 4: Generate uncertainty intervals
        # One slot per channel (current_data.shape[0] is the number of channels, not a
        # batch size), all starting at 0.
        uncertainty_intervals = torch.zeros(current_data.shape[0], device=current_data.device)

        # Go through the channels one by one. For a flagged channel, record the "width" of
        # its change: the largest minus the smallest cell-wise difference. For the others
        # record 0. Despite the name, this is not a statistical confidence interval; it is
        # just max - min of the changes in that channel.
        # Speed note: `.item()` and the `if` on a GPU tensor force a GPU -> CPU wait on
        # every iteration; with 256 channels this loop is likely slow (it could be
        # vectorised).
        for i in range(uncertainty_intervals.shape[0]):
            if uncertainty_flags[i]:
                # Example: Create interval based on score and threshold
                lower_bound = conformity_scores_whole[i].min().item()
                upper_bound = conformity_scores_whole[i].max().item()
                uncertainty_intervals[i] = upper_bound - lower_bound
            else:
                uncertainty_intervals[i] = 0.0  # No interval for conforming channels

        # return quantile_threshold, uncertainty_flags, uncertainty_intervals
        # Returns only the intervals, shape [C]: zeros for "confident" channels and a
        # positive number for "uncertain" ones. (The docstring's other return values were
        # removed.) These numbers are what the selector network receives.
        return uncertainty_intervals

    #CooperTrim autonet
    # CooperTrim step 1: the EGO car decides which channels to ask its neighbours for.
    # For each scene in the batch it produces a "request" (a list of channel numbers)
    # plus bookkeeping. The result is a list with one dictionary per scene.
    #
    # NOTE: the docstring below is partly stale (it mentions prev_avg_entropy, an older
    # idea). The arguments that matter are `epoch` (current training epoch),
    # `prev_fused_feature` (the previous frame's ego fused feature) and `validation`
    # (True when evaluating rather than training).
    def create_request(self, divided_batches, channel_select_model, epoch, prev_fused_feature=None, validation=False):
        """
        Process divided_batches to integrate orig_bev_data_from_all_cav and select BEV points.

        Args:
            divided_batches (list): List of tensors, each of shape (l, c, h, w), where l is the batch size.
            record_len (list): List of integers representing batch sizes.
            channel_select_model (callable): Model to predict mask for channel selection.
            prev_avg_entropy (float, optional): Previous average entropy for adaptive selection.

        Returns:
            list: Processed batches with selected BEV data integrated.
        """
        # One dictionary will be appended per scene.
        processed_batches = []
        for i, batch_data in enumerate(divided_batches):
            # Orig BEV data from all CAVs
            # The feature maps of all cars in this scene: shape [number of cars, C, H, W].
            orig_bev_data_from_all_cav = batch_data  # Assuming batch_data is the input BEV data

            # Process the first element (ego data) of the batch
            # The ego car's own feature map, shape [C, H, W]. (By convention car 0 is the ego.)
            data_at_index_0 = orig_bev_data_from_all_cav[0]  # Shape: (c, h, w)
            dim_len, height, width = data_at_index_0.shape  # Extract dimensions

            # Despite its name, this is the number of CHANNELS (256).
            num_spatial_indices = data_at_index_0.shape[0]  # Number of channels
            # `random_indices` will hold the numbers of the requested channels. The name is a
            # leftover from an earlier version that picked channels at random.
            random_indices = None
            select_threshold = None
            percentage_selected = None

            # Adaptive selection logic
            # # if prev_avg_entropy is not None:            
            # std_dev = data_at_index_0.std(dim=(1, 2))  # Compute standard deviation across spatial dimensions
            # std_dev = std_dev.unsqueeze(0)  # Add batch dimension for model input
            # # Predict mask using the channel selection model
            # predicted_mask, select_threshold = channel_select_model(std_dev, data_at_index_0.unsqueeze(0))
            # random_indices = predicted_mask.squeeze(0).long()  # Convert probabilities to binary mask
            # total_indices = random_indices.numel()
            # selected_indices = random_indices.sum().item()  # Count of selected indices
            # percentage_selected = (selected_indices / total_indices) * 100
            # print(f"Batch {i}: percentage_selected = {percentage_selected:.2f}%")
            # # else:
            # #     # Default selection logic (select all data)
            # #     percentage_data_to_request = 1.0  # Request 100% of data
            # #     num_random_indices = int(percentage_data_to_request * num_spatial_indices)
            # #     random_indices = torch.arange(num_spatial_indices, device=data_at_index_0.device)[:num_random_indices]
            # #     prev_avg_entropy = 1.0  # Initialize entropy
            # #     percentage_selected = 100.0
            # #     print(f"Batch {i}: percentage_selected = {percentage_selected:.2f}%")

            # (The large commented-out block above is an older experiment that used each
            # channel's standard deviation instead of the conformal uncertainty. It is dead code.)
            #
            # EPSILON-GREEDY EXPLORATION (paper Section 3.3). epsilon = 0.1 is the exploration
            # probability, hard-coded here.
            epsilon = 0.1  # Exploration probability
            # Use CooperTrim's selection only if ALL of these hold:
            #   * CooperTrim is switched on, and
            #   * a previous frame exists to compare against, and
            #   * either we are validating (always use the learned selection), or
            #     the epoch is 5 or later AND a random draw is above epsilon (90% of the time).
            # Otherwise (first 5 epochs, the 10% exploration draws, no previous frame, or
            # CooperTrim off) the ego requests ALL channels (the `else` branch below).
            # The paper describes the 10% full requests; the "first 5 epochs" part comes from
            # the code.
            if self.use_coopertrim and prev_fused_feature is not None and ((epoch >= 5 and random.random() > epsilon) or validation):
            # if epoch > 2 and random.random() > epsilon:  # Exploit with probability
            #         # Check if prev_fused_feature is provided
            #         if prev_fused_feature is not None:
                        # uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=50).unsqueeze(0)
                    #     # CooperTrim learnable confidence level
                    # Step A: how much did each channel change since the previous frame?
                    # (.unsqueeze(0) adds a batch dimension: [C] -> [1, C], as the selector expects.)
                    # NOTE: inside, the intervals are filled from Python numbers via `.item()`, so this
                    # tensor carries no gradient information.
                    uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=self.confidence_level).unsqueeze(0)
                        

                    # Step B: the selector turns those uncertainties (plus the ego's feature map) into
                    # a 0/1 mask over the 256 channels (1 = request), and also returns its threshold.
                    predicted_mask, select_threshold = self.channel_select_model(uncertainty_intervals, data_at_index_0.unsqueeze(0)) 
                    # Step C: convert the 0/1 mask into a list of channel numbers, e.g. [3, 17, 42].
                    # NOTE: this produces integer indices. Picking channels by number cannot pass a
                    # training signal back to the selector (see channel_select_attention.py).
                    indices = torch.where(predicted_mask[0] == 1)#.squeeze(1)
                    random_indices = indices[0]#.squeeze(0)

                    # Step D: the share of channels requested = requested / total * 100. This is the
                    # bandwidth measure that the loss penalises (about 100 means full sharing).
                    # NOTE: it is a plain Python number (a count divided by a total), not a tensor, so
                    # the penalty in the loss cannot push the selector to request fewer channels by
                    # ordinary back-propagation. (The print below runs for every scene.)
                    total_indices = predicted_mask.shape[1]
                    selected_indices = random_indices.shape[0]  # Count of selected indices
                    percentage_selected = (selected_indices / total_indices) * 100
                    print(f"percentage_selected: {(percentage_selected):.2f}%")

                    # #    print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected")          
            else:
                    # Fallback: request ALL channels (100%). Used in the first epochs, in the epsilon
                    # "explore" draws, when there is no previous frame, or when CooperTrim is off.
                    # (`self.prev_avg_entropy` below is a leftover of an older version; I found no use
                    # of it elsewhere in this file.)
                    percentage_data_to_request = 1.0
                    num_random_indices = int(percentage_data_to_request * num_spatial_indices)  # Compute 30% of total indices
                    #CooperTrim Transmission 1 - this data is transmitted from ego to CAV for request
                    # random_indices = torch.randperm(num_spatial_indices, device=flattened_data.device)[:num_random_indices]  # Random 30% indices
                    random_indices = torch.arange(num_spatial_indices, device=data_at_index_0.device)[:num_random_indices]
                    self.prev_avg_entropy = 1
                    percentage_selected = 100.0
                    print(f"percentage_selected: {(percentage_selected):.2f}%")

            # Integrate processed data into the batch
            # Everything later steps need for this scene:
            #  * orig_bev_data_from_all_cav : every car's full feature map (still local)
            #  * random_indices             : the requested channel numbers
            #  * select_threshold           : the selector's threshold (None in fallback mode)
            #  * percentage_selected        : the share of channels requested, for the loss
            processed_batch = {
                "orig_bev_data_from_all_cav": orig_bev_data_from_all_cav,#.to(divided_batches[0].device),
                "random_indices": random_indices, #.to(divided_batches[0].device),
                "select_threshold": select_threshold, #.to(divided_batches[0].device),
                "percentage_selected": percentage_selected,#.to(divided_batches[0].device),
            }
            processed_batches.append(processed_batch)

        # One dictionary per scene, in order.
        return processed_batches
    
    #CooperTrim autonet
    # CooperTrim step 2: the neighbours' reply. For every car in the scene (the ego
    # included), copy out ONLY the requested channels. In a real deployment only these
    # slices would travel over the wireless link; here everything sits in memory.
    #
    # NOTE: the docstring below is stale (it lists arguments that no longer exist).
    def create_response(self, processed_batches):
        """
        Integrates processed_batches into a new tensor with transformations and replication.

        Args:
            processed_batches (list): List of processed batches containing selected indices and BEV data.
            record_len (torch.Tensor): Tensor representing the number of records in each batch.
            max_cav (int): Maximum number of CAVs to consider.
            transformation_matrix (torch.Tensor): Transformation matrix for spatial alignment.
            sttf_function (callable): Function to apply spatial transformation.

        Returns:
            torch.Tensor: Integrated tensor after processing.
        """
        batch_size = len(processed_batches)
        integrated_tensor = None  # Placeholder for the final integrated tensor
        # One tensor per scene, holding just the requested channels.
        selected_output_values =[]
        for batch_idx, processed_batch in enumerate(processed_batches):
            orig_bev_data_from_all_cav = processed_batch["orig_bev_data_from_all_cav"]#.to(processed_batch["orig_bev_data_from_all_cav"].device)
            selected_indices = processed_batch["random_indices"]

            # # Step 1: Regroup data
            # x, _ = regroup(orig_bev_data_from_all_cav, record_len, max_cav)

            # # Step 3: Rearrange dimensions
            # x = rearrange(x, 'b l h w c -> b l c h w')

            # Extract dimensions
            n, c, h, w = orig_bev_data_from_all_cav.shape
            # batch_size = processed_batches.shape[0]

            # Step 4: Select output values based on selected indices
            # An empty container of shape [cars, number of requested channels, H, W]. No device
            # is given, so it is created on the CPU, and the loop below copies channel by
            # channel from the GPU. This probably costs speed (I have not profiled it).
            selected_output_values.append(torch.zeros(
                n, selected_indices.shape[0], h, w #, device=processed_batches.device
            ))
            # Copy requested channel number `value` of every car into slot `idx`.
            # (.clone() makes an independent copy. Gradients still flow back to the features
            # through these copies, which is how the detection loss trains the encoder.)
            for idx, value in enumerate(selected_indices):
                selected_output_values[-1][:, idx, :, :] = orig_bev_data_from_all_cav[:, value, :, :].clone()

        # List of tensors, one per scene.
        return selected_output_values
        
    # CooperTrim step 3: the ego car reassembles full-size feature maps from the
    # partial replies, so the fusion network gets the shape it expects.
    # For each car slot: start from a copy of the EGO car's own full feature map, then
    # overwrite the requested channels with that car's real values.
    # Result: requested channels hold each car's real data; all other channels hold the
    # ego's own data (so the fusion sees no empty or garbage channels).
    def accumulate_features_at_ego(self, processed_batches, selected_output_values, target_device ) :
        batch_size = len(processed_batches)
        integrated_tensor = None  # Placeholder for the final integrated tensor
        for batch_idx, processed_batch in enumerate(processed_batches):
            orig_bev_data_from_all_cav = processed_batch["orig_bev_data_from_all_cav"]
            # The requested channel numbers, moved to the same device (GPU) as the model.
            selected_indices = processed_batch["random_indices"].to(target_device)
            # Step 5: Replicate ego data across all CAVs
            cav_id_0_data = orig_bev_data_from_all_cav[0]
            n, c, h, w = orig_bev_data_from_all_cav.shape
            
            # Stack n copies of the ego map: [n, C, H, W]. `expand` creates a view that shares
            # memory, so the next lines call .clone() to make it writable.
            replicated_data = cav_id_0_data.unsqueeze(0).expand(n, -1, -1, -1)  # Shape: [n, c, h, w]
            # replicated_data = replicated_data.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # Shape: [1, n, c, h, w]
            # Step 6: Replace selected indices in replicated data
            selected_output_values_k = selected_output_values[batch_idx][:n, :, :, :].to(target_device)  # Select specific values
            replicated_data = replicated_data.clone().to(target_device)
            # Overwrite the requested channels in every car's copy with that car's real values
            # (this is the "received data").
            replicated_data[:, selected_indices, :, :] = selected_output_values_k[:, :len(selected_indices), :, :].to(replicated_data.dtype)
            # replicated_data = replicated_data.squeeze(0)
            # Update integrated tensor
            # Join the scenes back together along the first dimension: [total cars, C, H, W].
            if integrated_tensor is None:
                integrated_tensor = replicated_data#.unsqueeze(0)
            else:
                integrated_tensor = torch.cat((integrated_tensor, replicated_data), dim=0) #torch.cat((integrated_tensor, replicated_data.unsqueeze(0)), dim=0)

        # The reconstructed feature maps, in the same order as the original stacked tensor.
        return integrated_tensor

    # The full pass through the model for one batch.
    #  data_dict : the batch (LiDAR voxels, record_len, ...) prepared by the dataset
    #  epoch     : current training epoch (needed for CooperTrim's epsilon-greedy schedule)
    #  validation: True when evaluating, so CooperTrim always uses the learned selection
    # Returns (output_dict, select_thresholds, percentage_selected). The last two are
    # CooperTrim extras: train.py passes percentage_selected to the loss.
    def forward(self, data_dict, epoch, validation=False):
        # Unpack the pre-processed LiDAR data. The dataset has already grouped the points
        # into pillars ("voxels"): their features, their grid coordinates, and how many
        # points each contains.
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        # Number of cars in each scene of the batch (see the glossary).
        record_len = data_dict['record_len']
        # Read but never used in this file. (OpenCOOD's data loader typically transforms
        # other cars' point clouds into the ego car's coordinates beforehand; I did not
        # check this here.)
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # The encoder layers pass data around in one dictionary.
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        # PointPillars step 1: pillar points -> one vector per pillar. (n, 4) -> (n, 64)
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        # PointPillars step 2: place the pillar vectors on the 2D grid.
        batch_dict = self.scatter(batch_dict)
        # PointPillars step 3: the CNN backbone. The output is "spatial_features_2d".
        batch_dict = self.backbone(batch_dict)

        # The BEV feature maps of all cars in all scenes: [N, 384, H, W].
        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            # Reduce 384 -> 256 channels.
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        #CooperTrim autonet
        # print(f"spatial_features_2d shape = {spatial_features_2d.shape}")
        # ---- CooperTrim exchange (simulated) -------------------------------------------
        # Remember which device (GPU) the data is on, because the reply tensors are built
        # on the CPU.
        target_device = spatial_features_2d.device
        # Divide spatial_features_2d into batches
        # Split into scenes.
        divided_batches = self.divide_batches(spatial_features_2d, record_len)
        # Process divided_batches   
        # Ego decides what to request (uses the previous frame's fused feature).
        processed_batches = self.create_request(divided_batches, self.channel_select_model, epoch, self.prev_fused_feature, validation=validation)
        # Cars reply with only the requested channels.
        selected_output_values = self.create_response(processed_batches)
        # The ego rebuilds full-size maps for every car.
        reconstructed_data_at_ego = self.accumulate_features_at_ego( processed_batches, selected_output_values, target_device )
        # From here on, the rest of the model uses the reconstructed maps.
        spatial_features_2d = reconstructed_data_at_ego.clone()
        # print(f"spatial_features_2d shape after reconstruction= {spatial_features_2d.shape}")

    
        # compressor
        if self.compression:
            # Optional extra compression (disabled in the CooperTrim detection config).
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        # N, C, H, W -> B,  L, C, H, W
        # Rearrange [N, C, H, W] into [scene, car slot, C, H, W], padding scenes with fewer
        # than max_cav cars. `mask` says which slots hold a real car.
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # Reshape the mask to the feature-map size so the fusion network can ignore empty slots.
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])

        # CoBEVT fusion: merge all cars' features into one map per scene, shape [scenes, 256, H, W].
        fused_feature = self.fusion_net(regroup_feature, com_mask)
        #CooperTrim autonet
        # CooperTrim: remember the ego's fused map (of the FIRST scene in the batch only)
        # to serve as the reference for the next call. The config uses batch_size 1, so
        # that is the only scene. It is not .detach()ed, so it keeps its memory until
        # replaced on the next call.
        self.prev_fused_feature = fused_feature[0]  # Store ego fused feature for next iteration

        # Output 1: "is there a vehicle?" score map.
        psm = self.cls_head(fused_feature)
        # Output 2: box regression map.
        rm = self.reg_head(fused_feature)

        # The loss function reads these two entries.
        output_dict = {'psm': psm,
                       'rm': rm}

        #CooperTrim autonet
        # Initialize empty lists to store the values
        # Collect the CooperTrim bookkeeping from each scene to return alongside the outputs.
        select_thresholds = []
        percentage_selected = []

        # Iterate through the list and extract values
        for batch in processed_batches:
            select_thresholds.append(batch["select_threshold"])
            percentage_selected.append(batch["percentage_selected"])
        # Three return values; models in the OpenCOOD framework usually return just
        # output_dict. The training and inference scripts must unpack all three.
        return output_dict, select_thresholds, percentage_selected