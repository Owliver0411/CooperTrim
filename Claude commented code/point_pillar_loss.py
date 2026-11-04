# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
==============================================================================
FILE RECAP: point_pillar_loss.py   (3D_Detection_OPV2V version)
Original location: 3D_Detection_OPV2V/opencood/loss/
==============================================================================

WHAT THIS FILE IS
-----------------
The "loss function" for the detection model. A loss is a single number that
measures how wrong the model's output is. Training means repeatedly adjusting
the model's weights so this number gets smaller. The config line
`loss: core_method: point_pillar_loss` selects this file, and tools/train.py
calls it once per batch.

WHAT THE LOSS IS MADE OF
------------------------
   total_loss = reg_loss + conf_loss + select_channel_wt * percentage_selected
                 ^^^^^^     ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 box accuracy   |            |         CooperTrim's bandwidth penalty
                |            "is there a vehicle?" accuracy
 1. conf_loss : classification, measured with "focal loss" (cls_loss_func).
 2. reg_loss  : regression, how far the predicted boxes are from the true boxes
                (smooth L1 loss; the angle gets a special treatment).
 3. CooperTrim: a penalty that grows with the share of channels requested, so the
                model is "charged" for using bandwidth (paper Section 3.3, Eq. 2).

WHICH PARTS ARE COOPERTRIM'S
----------------------------
The header credits OpenPCDet and Runsheng Xu (the OpenCOOD author), so almost
everything here is existing code. CooperTrim's changes are the lines marked
"#CooperTrim autonet": the `select_channel_wt` setting in __init__, the extra
arguments of forward(), and the block that builds the bandwidth penalty.

GLOSSARY
--------
 Anchor       : a template box placed at every grid cell. The model scores each
                one ("vehicle here?") and predicts a correction for its box.
 Positive     : an anchor that matches a real vehicle. Negative: one that does not.
 Logit        : a raw model score before it is squashed into 0..1 with a sigmoid.
 Epoch        : one full pass over the training data.

THINGS WORTH KNOWING (my reading of the code; I have not run it)
----------------------------------------------------------------
 1. `percentage_selected` reaches this file as a plain Python number, not a
    tensor. A plain number has no connection to the model's weights, so adding
    `select_channel_wt * percentage_selected` raises the reported loss but gives
    the training no direction to follow. The penalty therefore appears not to
    influence the selector.
 2. The penalty weight follows a schedule (see the block in forward()). It resembles
    Appendix A.1 of the paper, but here each epoch's weight is recomputed from the
    base weight rather than multiplying the previous one.
 3. `select_channel_wt` is not set in the shipped detection config, so it takes the
    default 0.01.
 4. Unlike the segmentation loss, the detection version does not subtract a target
    (that one uses `percentage_selected - 4.0`), so the paper's "P - C" form is not
    literal here. Since the term is constant for the optimiser it makes no
    difference to the gradients either way.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Smooth L1 loss, used for box regression. In words: for a small error, the penalty
# is the error squared (so it is gentle near zero); for a large error, the penalty
# grows only in proportion to the error (so a few wild mistakes cannot dominate).
# This part is existing code (modified from fvcore).
class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    # beta: the error size at which the loss switches from "squared" to "linear".
    # code_weights: optional importance per predicted number (unused in this repo).
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            # Only runs if code_weights is given; `.cuda()` moves it to the GPU.
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            # n = absolute error. If n < beta use the squared form, otherwise the linear form
            # (the two formulas meet smoothly at n = beta).
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    # Computes the loss per anchor and per predicted number (no averaging yet).
    # `input` = model predictions, `target` = true values, `weights` = how much each
    # anchor counts.
    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        # Where the true value is missing (NaN), pretend the target equals the prediction so
        # the difference is zero and that entry is ignored.
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        # The error for every predicted number.
        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        # Multiply each anchor's loss by its weight. In this file the weights are 1/(number
        # of positives) for positive anchors and 0 for the rest, so only anchors that match a
        # real vehicle contribute to the box loss.
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


# The loss used for training. Calling it like a function runs forward().
class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        # The box-regression loss defined above.
        self.reg_loss_func = WeightedSmoothL1Loss()
        # Focal-loss settings (see cls_loss_func): alpha balances positives against
        # negatives, gamma (next line) makes the loss focus on the hard examples.
        self.alpha = 0.25
        self.gamma = 2.0

        # Importance factors read from the YAML `loss: args:` section (cls_weight is 1.0
        # and reg is 2.0 in the CooperTrim detection config): how much the classification
        # and the box loss count in the total.
        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        # Stores the latest values so logging() can print them.
        self.loss_dict = {}

        #CooperTrim autonet
        # CooperTrim: the base weight of the bandwidth penalty. If the config does not
        # provide `select_channel_wt` (it does not here), the default is 0.01.
        self.select_channel_wt = args['select_channel_wt'] if 'select_channel_wt' in args else 0.01

    #CooperTrim autonet
    # Computes the total loss for one batch.
    #  output_dict        : the model outputs ('psm' scores and 'rm' box corrections)
    #  target_dict        : the ground truth (true boxes and which anchors are positive)
    #  percentage_selected: CooperTrim extra: the share of channels the ego requested
    #                       (a list with one value per scene, or a single number)
    #  epoch              : CooperTrim extra: current epoch, used for the penalty schedule
    # Returns the single total loss value.
    def forward(self, output_dict, target_dict, percentage_selected, epoch):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        # The two model outputs, and the true box targets.
        rm = output_dict['rm']
        psm = output_dict['psm']
        targets = target_dict['targets']

        # Reorder the scores from [B, anchors, H, W] to [B, H, W, anchors] and flatten later.
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        # Ground truth per anchor: 1 where the anchor matches a real vehicle, 0 elsewhere.
        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        # Split anchors into positives (real vehicle) and negatives (empty road).
        # Classification counts every anchor; regression counts only the positives.
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        # Divide by the number of positives (at least 1) so scenes with many vehicles do not
        # dominate those with few.
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        # Build the classification targets: one-hot over 2 classes (0 = background,
        # 1 = vehicle), then keep only the "vehicle" column (`[..., 1:]` below), giving one
        # yes/no target per anchor.
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        # Focal loss for each anchor (defined in cls_loss_func below).
        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        # Add up over all anchors, average over the batch, then apply cls_weight (next line).
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # Box regression: compare predicted boxes with the true boxes.
        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        # Convert the heading angles with the "sin trick" (see add_sin_difference).
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        # Add up, average over the batch, then multiply by the box-loss importance (`reg`, next line).
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # total_loss = reg_loss + conf_loss
        
        #CooperTrim autonet
        # #l5
        # ---- CooperTrim: bandwidth penalty weight (paper Section 3.3 and Appendix A.1) ----
        # The penalty is switched on gradually:
        #   * epochs 0-5 : weight 0 (learn detection first, with no bandwidth pressure);
        #                  "ft" stands for fine-tuning.
        #   * every 10th epoch (10, 20, ...): weight = base * 2^(share/100), i.e. between
        #                  base and 2 * base, larger when many channels are requested.
        #   * other epochs: weight = base * (1 + 0.1 * number of completed 10-epoch blocks).
        # `base` is self.select_channel_wt (0.01 by default).
        ft_epoch_count = 5 #20
        # Warm-up phase: no penalty.
        if epoch <= ft_epoch_count:
            select_channel_wt = 0.0        
        # Every 10th epoch (after the warm-up).
        elif epoch % 10 == 0:
            # Dynamically scale the weight based on the percentage_selected
            # Scale select_channel_wt exponentially or linearly based on percentage_selected
            # Handle different types of percentage_selected
            if isinstance(percentage_selected, list):
                percentage_selected = percentage_selected[0]  # Take the first element if it's a list
            # If it is a list (one value per scene) only the first scene's value is used.
            # The share is converted to a 0-1 fraction first (e.g. 32% -> 0.32).
            scaling_factor = percentage_selected / 100.0  # Normalize percentage to [0, 1]

            # scaling_factor = percentage_selected / 100.0  # Normalize percentage to [0, 1]
            select_channel_wt = self.select_channel_wt * (2 ** scaling_factor)  # Exponential scaling
        else:
          
            # Other epochs: grow the weight by 10% of the base for every 10 epochs since the warm-up (`//` is whole-number division).
            select_channel_wt = self.select_channel_wt * (1 + 0.1 * ((epoch-ft_epoch_count) // 10))  # Linear growth
        print(f"Epoch: {epoch}, Select Channel Weight: {select_channel_wt}")
        if isinstance(percentage_selected, list):
                percentage_selected = percentage_selected[0]
        # avg_percentage_selected = sum(percentage_selected) / len(percentage_selected)
        # The total loss: box loss + classification loss + CooperTrim penalty.
        # NOTE: `percentage_selected` is a plain Python number (see the file header), so the
        # last term adds a constant to the loss value without giving the optimiser any
        # direction. Only reg_loss and conf_loss steer the training.
        total_loss = reg_loss + conf_loss + select_channel_wt * percentage_selected

        # Save the values for logging().
        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss})

        return total_loss

    # Focal loss for the classification of anchors. Most anchors are empty road and
    # easy to get right. Focal loss shrinks the contribution of easy anchors so the rare,
    # hard ones (real vehicles) dominate the learning. Existing code, not CooperTrim.
    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        # Squash the raw scores into probabilities between 0 and 1.
        pred_sigmoid = torch.sigmoid(input)
        # alpha weighting: positives get 0.25, negatives 0.75 (balances the classes).
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        # pt = how wrong the prediction is (near 0 when confident and right).
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        # Raising pt to the power gamma (2) makes already-correct anchors count much less.
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        # The standard "binary cross-entropy" error for each anchor.
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        # Combine: error x focal weight.
        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        # A numerically stable way to compute binary cross-entropy directly from raw scores
        # (same formula as TensorFlow's sigmoid_cross_entropy_with_logits).
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        # Angle handling for the box heading (the 7th number, index 6). Angles wrap around
        # (359 degrees is almost the same as 1 degree), so subtracting them directly would
        # mislead the loss. With the products below, the difference between the two encodings
        # equals sin(predicted angle - true angle): zero when the angles agree, and no jump at
        # the wrap-around. (It is also zero when the angles differ by 180 degrees, so a
        # box facing exactly backwards is not penalised by this term.)
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    # Prints the loss values during training and writes them to TensorBoard (a viewer for
    # training curves). Only the box and classification parts are logged here, not the
    # bandwidth penalty.
    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item(), reg_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item(), reg_loss.item()))


        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)