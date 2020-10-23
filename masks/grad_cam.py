#Source from https://github.com/maxs-kan/InterpretableNeuroDL
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
    
class GuidedBackprop():
    def __init__(self, model):
        self.model = model
    
    def guided_backprop(self, input, label):
        
        def hookfunc(module, gradInput, gradOutput):
            return tuple([(None if g is None else g.clamp(min=0)) for g in gradInput])
    
        input.requires_grad = True
        h = [0] * len(list(self.model.features) + list(self.model.classifier))
        for i, module in enumerate(list(self.model.features) + list(self.model.classifier)):
            if type(module) == nn.ReLU:
                h[i] = module.register_backward_hook(hookfunc)

        self.model.eval()
        output = self.model(input)
        self.model.zero_grad()
        output[0][label].backward()
        grad = input.grad.data
        grad /= grad.max()
        return np.clip(grad.cpu().numpy(),0,1)


class AttentionMap():
    def __init__(self, model):
        self.model = model
    
    def guided_backprop(self, input):
        input.requires_grad = True
        self.model.eval()
        act = self.model.get_activations(input)
        attention = []
        for c in range(act.shape[1]):
            activation = self.model.get_activations(input)
            activation[0, c].backward(torch.ones_like(activation[0, c]))
            attention.append(input.grad.data.cpu().numpy().squeeze(0))
            self.model.zero_grad()
        return np.concatenate(attention, axis=0)
    
def get_masks(model, loader, device, mask_type='grad_cam'):
    masks = []
    gp = GuidedBackprop(model)
    for image, gt in tqdm(loader, total=len(loader)):
        image = image.to(device)
        logit = model(image)
        if mask_type == 'grad_cam':
            logit[:,1].backward()
            activation = model.get_activations(image).detach()
            act_grad  = model.get_activations_gradient()
            pool_act_grad = torch.mean(act_grad, dim=[2,3,4], keepdim=True)
            activation = activation * pool_act_grad
            heatmap = torch.sum(activation, dim=1)
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)
            heatmap = F.interpolate(heatmap.unsqueeze(0),(180,180,180), mode='trilinear', align_corners=False)
            masks.append(heatmap.cpu().numpy())
        
        elif mask_type == 'guided_backprop':
            pred = logit.data.max(1)[1].item()
            img_grad = gp.guided_backprop(image, pred)
            masks.append(img_grad)
        else:
            raise NotImplementedType('define mask_type')
            
    return np.concatenate(masks,axis=0).squeeze(axis=1)
        

