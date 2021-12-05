import torch



""" implementation of student-t distribution """
def studentT_fct(distances, theta_boundary, device='cpu'):
    torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

    #print("theta",theta_boundary, theta_boundary.shape)
    prefactor = 1 / (torch.pi * theta_boundary.to(device))
    #print("prefa",prefactor, prefactor.shape)

    #print("distances",distances)
    distribution = 1 / (1 + (distances.to(device) / (theta_boundary.to(device) ** 2)))

    studentT = prefactor * distribution

    return studentT

def studentT(distances, theta_boundary, idx=None, prototype_labels=None, device='cpu'): 
    # probabilitys of heavy tailed fct
    probs = studentT_fct(distances, theta_boundary, device=device)
    
    # normalize
    zero = torch.Tensor([[0]])
    norm_scalar = studentT_fct(zero, theta_boundary)
    probs = probs / norm_scalar
  
    #print(studentT_fct(theta_boundary, theta_boundary)/norm_scalar)

    if type(idx) == torch.Tensor:
        #print("probs_normed:",probs)
        #print(idx)
        winning_indices = idx
        winning_indices = torch.where(
                winning_indices <= torch.amax(prototype_labels),
                winning_indices, 
                torch.min(distances, dim=1).indices).to(device)
    else:
        # winning indices of prototypes
        winning_indices = torch.min(distances, dim=1).indices.to(device) # list of winning prototypes

    #studentT = distribution
    probs = probs.gather(1, winning_indices.unsqueeze(1)).squeeze()
    #print("gathered:",probs)

    return probs

