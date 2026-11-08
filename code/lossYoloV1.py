import torch
import torch.nn as nn

def intersection_over_union(boxes_preds, boxes_labels, is_yolo:bool=True):
    """
    Calculates Intersection over Union (IoU) with support for broadcasting.
    
    Args:
        boxes_preds (Tensor): Predicted boxes, shape [..., num_boxes, 4]
        boxes_labels (Tensor): Target boxes, shape [..., 1, 4]
        box_format (str): 'midpoint' (x,y,w,h) or 'corners' (x1,y1,x2,y2)
        
    Returns:
        Tensor: IoU for all pairs, shape [..., num_boxes]
    """
    if is_yolo:
        pred_w = boxes_preds[..., 2:3].clamp(min=1e-6)
        pred_h = boxes_preds[..., 3:4].clamp(min=1e-6)

        target_w = boxes_labels[..., 2:3].clamp(min=1e-6)
        target_h = boxes_labels[..., 3:4].clamp(min=1e-6)

        box1_x1 = boxes_preds[..., 0:1] - pred_w / 2
        box1_y1 = boxes_preds[..., 1:2] - pred_h / 2
        box1_x2 = boxes_preds[..., 0:1] + pred_w / 2
        box1_y2 = boxes_preds[..., 1:2] + pred_h / 2

        box2_x1 = boxes_labels[..., 0:1] - target_w / 2
        box2_y1 = boxes_labels[..., 1:2] - target_h / 2
        box2_x2 = boxes_labels[..., 0:1] + target_w / 2
        box2_y2 = boxes_labels[..., 1:2] + target_h / 2

    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # 1. Calculate intersection coordinates using element-wise max/min
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 2. Compute intersection area (.clamp(0) handles cases with zero overlap)
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # 3. Compute individual box areas
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # 4. Calculate union area (Area A + Area B - Intersection)
    union = box1_area + box2_area - intersection

    # 5. Return IoU score (1e-6 prevents dividing by absolute zero)
    return (intersection / (union + 1e-6)).squeeze(-1)



class YoloV1Loss(nn.Module):
    def __init__(self, S:int=7, B:int=2, C:int=1):
        super(YoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mse = nn.MSELoss(reduction="sum")
        self.lam_no_obj = 0.75
        self.lam_obj = 5
    def forward(self, predictions:torch.Tensor, target:torch.Tensor):
        """
        predictions will have (num_batches, S*S*(B*5 + C))
        target will have (num_batches, S, S, C + 1*5)
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        start_index = self.C + self.B
        all_pred_boxes = predictions[..., start_index:(start_index + self.B*4)].reshape(*predictions.shape[:-1], self.B, 4)
        target_box = target[..., (self.C+1):(self.C+5)].unsqueeze(-2)
        ious = intersection_over_union(all_pred_boxes, target_box)
        ious_max = torch.argmax(ious, dim=-1)
        best_box_expanded = ious_max.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 4)

        # For torch.gather to work the index must have the same shape as input
        exists_prob = target[..., self.C]
        exists_box = target[..., self.C].unsqueeze(-1).unsqueeze(-1)
        best_pred_box = torch.gather(all_pred_boxes, dim=-2, index=best_box_expanded)

        # To calculate box loss
        # 2:4 height and width
        # Use box_loss = (sqrt(sqrt(w1) - sqrt(w2))^2 + (sqrt(h1) - sqrt(h2))^2)
        pred_xy = best_pred_box[..., 0:2]
        pred_hw = torch.sqrt(
            torch.clamp(best_pred_box[..., 2:4], min=0) + 1e-6
        )
        transformed_pred = torch.cat([pred_xy, pred_hw], dim=-1)
        
        target_xy = target_box[..., 0:2]
        target_wh = torch.sqrt(target_box[..., 2:4])
        transformed_targets = torch.cat([target_xy, target_wh], dim=-1)

        box_predictions = exists_box*transformed_pred
        box_target = exists_box*transformed_targets

        box_loss = self.mse(
            torch.flatten(box_predictions),
            torch.flatten(box_target),
        )

        # To calculate probability of box

        all_pred_prob = predictions[..., (self.C):(self.C + self.B)].reshape(*predictions.shape[:-1], self.B) # shape is [batch, S, S, B]
        # Calculate IoU target for the responsible bounding box
        best_iou = torch.gather(
            ious,
            dim=-1,
            index=ious_max.unsqueeze(-1)
        ).detach()

        # Confidence prediction
        best_pred_prob = torch.gather(
            all_pred_prob,
            dim=-1,
            index=ious_max.unsqueeze(-1)
        )


        # Confidence target = IoU
        target_prob = best_iou

        object_loss = self.mse(
            torch.flatten(exists_prob.unsqueeze(-1) * best_pred_prob),
            torch.flatten(exists_prob.unsqueeze(-1) * target_prob)
        )

        # To calculate no object loss

        no_object_target = torch.zeros_like(all_pred_prob)

        responsible_box_mask = nn.functional.one_hot(ious_max, num_classes=self.B)
        no_object_mask = 1- exists_prob.unsqueeze(-1)*responsible_box_mask

        masked_object_pred = no_object_mask*all_pred_prob
        masked_object_target = no_object_mask*no_object_target

        no_object_loss = self.mse(
            torch.flatten(masked_object_pred),
            torch.flatten(masked_object_target)
        )

        # To calculate Class Loss

        class_pred = predictions[..., :self.C]
        class_target = target[..., :self.C]

        exists_prob = target[..., self.C].unsqueeze(-1)

        class_loss = self.mse(
            exists_prob * class_pred,
            exists_prob * class_target
        )
        
        # TOTAL LOSS
        loss = (
            self.lam_obj * box_loss
            + object_loss
            + self.lam_no_obj * no_object_loss
            + class_loss
        )

        individual_losses = torch.stack([
            box_loss,
            object_loss,
            no_object_loss,
            class_loss
        ])
        return loss, individual_losses