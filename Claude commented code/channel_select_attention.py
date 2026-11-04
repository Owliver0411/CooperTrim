"""
==============================================================================
FILE RECAP: channel_select_attention.py   (3D_Detection_OPV2V version)
Original location: 3D_Detection_OPV2V/opencood/models/sub_modules/
==============================================================================

THE BIG PICTURE
---------------
In "cooperative perception", several cars (called CAVs) each look at the road
with their own sensors and SHARE what they see so every car gets a better
picture than it could alone.  Cars do not share raw sensor data (too big);
they share a "feature map": a stack of grids of numbers produced by a neural
network.

Sending a full feature map every frame uses a lot of wireless bandwidth.
CooperTrim's idea: the car that is receiving ("the ego car") only ASKS its
neighbours for the parts it actually needs, chosen using UNCERTAINTY (how
unsure the ego car is about each part).

WHAT THIS FILE DOES
-------------------
It defines ONE small neural network: `CrossAttentionMaskPredictorAdaptive`.
Given
   (a) a per-channel uncertainty score for the ego car, and
   (b) the ego car's current feature map,
it outputs a "mask": one 0 or 1 for each channel of the feature map.
   1 = "yes, request this channel from the neighbours"
   0 = "no, skip it, save bandwidth"

It also returns a learnable `threshold` (the cut-off used to turn
probabilities into 0/1).

WHO USES IT
-----------
models/point_pillar_cobevt.py  ->  method `create_request` builds this module
and calls it each frame.  (The other two sub-projects, V2V4Real detection and
OPV2V segmentation, contain their own slightly different copies of this file.)

GLOSSARY (for readers new to deep learning)
-------------------------------------------
* Tensor     : a multi-dimensional array of numbers (PyTorch's core data type).
* Shape      : the size of each dimension, e.g. [1, 256, 96, 352].
* Batch (B)  : how many independent samples are processed at once.
* Channel (C): one "layer" of a feature map. A feature map of shape
               [B, C, H, W] has C separate H x W grids stacked together, each
               one capturing a different kind of pattern. CooperTrim chooses
               which CHANNELS to request.
* H, W       : height and width of the grid (spatial size).
* nn.Module  : PyTorch's base class for any neural network component.
* nn.Linear  : a layer that multiplies its input by a matrix of learnable
               numbers ("weights") and adds a bias. It is how a network
               learns to transform data.
* Attention  : a mechanism that learns "which things should I pay most
               attention to?" by comparing a QUERY against many KEYS and
               using the match scores to take a weighted average of VALUES.
* Softmax    : turns any list of numbers into positive numbers summing to 1
               (like percentages).
* Sigmoid    : squashes any number into the range 0 to 1 (like a probability).
* nn.Parameter: a tensor that PyTorch treats as learnable during training.
"""

# torch is PyTorch, the deep-learning library everything in this repo uses.
import torch
# `nn` contains the building blocks of neural networks (Linear layers, Module...).
import torch.nn as nn
# `F` contains plain functions such as softmax that do not hold learnable weights.
import torch.nn.functional as F



# A class that inherits from nn.Module is a neural-network component. PyTorch
# can then automatically find its learnable weights, move it to the GPU,
# save/load it, etc.
class CrossAttentionMaskPredictorAdaptive(nn.Module):
    def __init__(self, num_channels, spatial_dim_h, spatial_dim_w):
        """
        Cross-attention based mask predictor.
        Args:
        - num_channels: Number of input channels (e.g., 128).
        - spatial_dim: Spatial dimensions of the feature maps (e.g., 32x32).
        """
        # [NOTE] The docstring above is left over from an older version: there
        # is no longer a single `spatial_dim`; the grid size is now given as a
        # height (spatial_dim_h) and width (spatial_dim_w) separately.
        # In point_pillar_cobevt.py this class is built with
        # num_channels=256, spatial_dim_h=96, spatial_dim_w=352.

        # Run the parent class (nn.Module) setup. Required in every PyTorch
        # module, otherwise PyTorch cannot track the learnable weights below.
        super(CrossAttentionMaskPredictorAdaptive, self).__init__()

        # Remember the sizes so forward() and other code can reuse them.
        self.num_channels = num_channels
        self.spatial_dim_h = spatial_dim_h
        self.spatial_dim_w = spatial_dim_w

        # ---- Linear layers for query, key, and value projections ----------
        # Attention works with three ingredients: Query, Key, Value.
        #
        # QUERY = "what am I looking for?"  Built from the uncertainty scores
        # (one number per channel), so its size is num_channels.
        # nn.Linear(a, b) maps a vector of length a to a vector of length b.
        self.query_proj = nn.Linear(num_channels, num_channels)

        # KEYS = "what does each channel offer?"
        # Each channel is an H x W grid. Flattened, that is H*W numbers
        # (here 96 * 352 = 33,792). This layer compresses those H*W numbers
        # into a short vector of length num_channels (256), one per channel.
        # [NOTE] This is the biggest layer: 33,792 x 256 is ~8.6 million
        # weights, and `value_proj` below is the same size.
        self.key_proj = nn.Linear(spatial_dim_h * spatial_dim_w, num_channels)

        # VALUES = "the information to pass on if this channel gets attention".
        # Same shape and idea as the keys, but with its own separate weights.
        self.value_proj = nn.Linear(spatial_dim_h * spatial_dim_w, num_channels)

        # ---- Output layer to predict the mask ------------------------------
        # Takes the attention result (length num_channels) and produces one
        # score per channel (length num_channels).
        self.output_layer = nn.Linear(num_channels, num_channels)
        # Sigmoid squashes each score into 0..1 so it reads as "probability
        # that this channel should be requested".
        self.sigmoid = nn.Sigmoid()  # For binary mask probabilities

        #CooperTrim adapt learn parameter
        # A single learnable number (starts at 0.5). nn.Parameter tells PyTorch
        # "this is a trainable weight". In forward() it is passed through a
        # sigmoid to become the cut-off between "request" (1) and "skip" (0).
        # The goal: let training decide how strict the selection should be,
        # instead of hard-coding 0.5.
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Initialize threshold at 0.5

    def forward(self, std_dev, features):
        """
        Forward pass for the model.
        Args:
        - std_dev: Tensor of shape [batch_size, num_channels] (std dev of each channel).
        - features: Tensor of shape [batch_size, num_channels, spatial_dim, spatial_dim].
        Returns:
        - mask: Predicted binary mask probabilities [batch_size, num_channels].
        """
        # `forward` is what runs when you call the module like a function:
        #     mask, threshold = module(uncertainty, features)
        #
        # [NOTE] The argument is called `std_dev` because an earlier version
        # fed in the standard deviation of each channel. In the current model
        # code (point_pillar_cobevt.py) it is given `uncertainty_intervals`
        # from `compute_conformal_uncertainty`, i.e. still one uncertainty
        # number per channel. Also, the function returns TWO things
        # (mask, threshold), not just the mask as the docstring says.
        #
        # In the real model, `features` is the ego car's (index 0) feature map
        # with a batch dimension added: shape [1, 256, 96, 352].

        # Read the sizes off the input so the code works for any input size.
        # (The H and W here must match what the layers above were built with,
        #  or key_proj / value_proj will fail with a shape mismatch.)
        batch_size, num_channels, spatial_dim_h, spatial_dim_w = features.shape

        # Flatten spatial dimensions of the features: [batch_size, num_channels, spatial_dim*spatial_dim]
        # Turn each H x W grid into one long row of H*W numbers.
        #   [B, C, H, W]  ->  [B, C, H*W]
        # (-1 means "work out this size yourself".)
        features_flat = features.view(batch_size, num_channels, -1)

        # Project standard deviation (queries): [batch_size, num_channels]
        # Pass the uncertainty vector through query_proj to get the QUERY.
        queries = self.query_proj(std_dev)  # Shape: [batch_size, num_channels]

        # Project features to keys and values: [batch_size, num_channels, num_channels]
        # Each channel's H*W numbers become a vector of length C, giving one
        # key and one value per channel:  [B, C, H*W] -> [B, C, C]
        # (The first "C" is "which channel", the second "C" is the vector.)
        keys = self.key_proj(features_flat)  # Shape: [batch_size, num_channels, num_channels]
        values = self.value_proj(features_flat)  # Shape: [batch_size, num_channels, num_channels]

        # Compute attention scores: [batch_size, num_channels]
        # einsum is a compact way of writing a multiply-and-sum. Reading
        # 'bn,bmn->bm': for every batch item b and every channel m, multiply
        # the query (length n) with that channel's key (length n) and add up.
        # That is a dot product: a big number means "this channel's key
        # matches what the uncertainty query is looking for".
        # Result: one score per channel.
        attention_scores = torch.einsum('bn,bmn->bm', queries, keys)  # Batch matrix multiplication
        # Softmax across the channels turns the raw scores into weights that
        # add up to 1 (a "how much attention each channel gets" list).
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores

        # Compute attention-weighted values: [batch_size, num_channels]
        # 'bm,bmn->bn': multiply each channel's value vector by its attention
        # weight and add them all up. The result, `context`, is a single
        # summary vector of length C that blends the channels, giving more
        # influence to the ones that received more attention.
        context = torch.einsum('bm,bmn->bn', attention_scores, values)

        # Predict mask using the context: [batch_size, num_channels]
        # Turn the summary into one score per channel...
        mask_logits = self.output_layer(context)  # Shape: [batch_size, num_channels]
        # ...and squash each into 0..1 -> "probability of requesting channel i".
        mask = self.sigmoid(mask_logits)  # Apply sigmoid for probabilities

        #CooperTrim adapt learn parameter
        # Turn the learnable raw number into a proper 0..1 cut-off. With the
        # starting value 0.5 this is sigmoid(0.5) ~= 0.62.
        l_threshold = torch.sigmoid(self.threshold)
        # Compare every probability with the cut-off:
        #   probability > threshold -> True -> 1.0  (request this channel)
        #   otherwise               -> False -> 0.0 (skip it)
        # `.float()` converts True/False into 1.0/0.0 so it can be used as a mask.
        # [NOTE / QUESTION] A "greater than" comparison is a hard step: PyTorch
        # cannot pass a training signal (gradient) back through it. So the
        # layers above and `self.threshold` may not be trained by ordinary
        # back-propagation through this line. I will check in
        # point_pillar_cobevt.py and the loss how they are actually trained.
        mask = (mask > l_threshold).float()
        
        # Returns two things:
        #   mask        : tensor [batch, channels] of 0.0 / 1.0 values
        #   l_threshold : the current cut-off value (a single number)
        return mask, l_threshold  


# Example Usage
# This block only runs when you execute THIS file directly
# (`python channel_select_attention.py`), never when another file imports it.
# It is a small demo with fake random data.
# [NOTE] As written, this demo is out of date and would crash:
#   1. The constructor now needs spatial_dim_h and spatial_dim_w, but the demo
#      passes `spatial_dim=...`.
#   2. forward() returns a (mask, threshold) tuple, but the demo treats the
#      result as one tensor.
# I have not changed the code, only commented it.
if __name__ == "__main__":
    # Example Data
    batch_size = 1
    num_channels = 128
    spatial_dim = 32

    # Random input data of shape [batch_size, 128, 32, 32]
    # randn = random numbers from a bell curve (a stand-in for a feature map).
    input_data = torch.randn(batch_size, num_channels, spatial_dim, spatial_dim)

    # Step 1: Compute Standard Deviation for Each Channel
    # std over the two spatial dimensions (H and W) gives one "spread" number
    # per channel: a simple stand-in for uncertainty.
    std_dev = input_data.std(dim=(2, 3))  # Shape: [batch_size, 128]

    # Step 2: Initialize Model
    model = CrossAttentionMaskPredictorAdaptive(num_channels=num_channels, spatial_dim=spatial_dim)

    # Step 3: Forward Pass
    predicted_mask = model(std_dev, input_data)  # Shape: [batch_size, 128]

    # Step 4: Convert Probabilities to Binary Mask
    # (Leftover from the older version whose output was probabilities. The
    # current forward() already returns 0/1 values.)
    binary_mask = (predicted_mask > 0.5).float()  # Threshold at 0.5

    # Print Results
    print("Input Data Shape:", input_data.shape)
    print("Standard Deviation Shape:", std_dev.shape)
    print("Predicted Mask Shape:", predicted_mask.shape)
    print("Binary Mask Shape:", binary_mask.shape)

    # Example Output
    print("Predicted Mask (Probabilities):", predicted_mask)
    print("Binary Mask:", binary_mask)
