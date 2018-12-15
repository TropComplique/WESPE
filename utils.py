import torch


def gradient_penalty(x, y, f):
    """
    Arguments:
        x, y: float tensors with shape [b, d, h, w].
        f: a pytorch module.
    Returns:
        a float tensor with shape []. 
    """
    
    # interpolation
    b = x.size(0)
    alpha = torch.rand([b, 1, 1, 1]).to(x.device)
    z = x + alpha * (y - x)
    z.requires_grad = True

    # compute gradient
    ones = torch.ones_like(z)
    g = grad(f(z), z, grad_outputs=ones, create_graph=True, only_inputs=True)[0]
    # it has shape [b, d, h, w]
    
    g = g.view(b, -1)
    return ((g.norm(p=2, dim=1) - 1.0)**2).mean()