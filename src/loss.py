import torch
import torch.nn.functional as F

def MSE_LOSS(rgb, rgb0, target):
    loss = F.mse_loss(rgb, target)
    loss0 = F.mse_loss(rgb0, target)
    return loss + loss0

def Adaptive_MSE_LOSS22(rgb, rgb0, target):

    non_zero_positions = (target != 0)

    if target.numel() == 0:
        return torch.tensor(0.0)

    target=target[non_zero_positions]
    
    # Get the corresponding values in output
    rgb=rgb[non_zero_positions]
    
    rgb0=rgb0[non_zero_positions]

    
    weights = torch.sqrt(torch.abs(rgb - target)).detach()
    loss = F.mse_loss(rgb, target)
    loss0 = torch.mean(weights * torch.square(rgb0 - target))
    return loss + loss0


#def Adaptive_MSE_LOSS(rgb, rgb0, target, gts_processed):

    #loss1 = l2_loss(rgb0, gts_processed)
    #loss2 = l2_loss(rgb, gts_processed)
    #target[gts_processed==0]=0
    
    ##weights = torch.sqrt(torch.abs(rgb - target)).detach()
    ##loss = F.mse_loss(rgb, target)
    ##loss0 = torch.mean(weights * torch.square(rgb0 - target))
    #loss0= Adaptive_MSE_LOSS22(rgb, rgb0, target)
    #return loss0 + 2*(loss1 + loss2)


#def Adaptive_MSE_LOSS(rgb0, target):
    
  
    #loss0 = torch.mean(torch.square(rgb0 - target))
    #return loss0


def Adaptive_MSE_LOSS(rgb, rgb0, target):

    weights = torch.sqrt(torch.abs(rgb - target)).detach()
    loss = F.mse_loss(rgb, target)
    loss0 = torch.mean(weights * torch.square(rgb0 - target))
    return loss + loss0


def l2_loss(output, gt):
    
    # Find positions where gt is 0
    zero_positions = (gt == 0)
    
    # Get the corresponding values in output
    output_values = output[zero_positions]
    
    # Calculate the L2 loss (mean squared error)
    loss = torch.mean(output_values ** 2)
    
    return loss

    