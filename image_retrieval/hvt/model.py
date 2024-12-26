import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import hyptorch.nn as hypnn

def init_model(cfg):
    """
    Initialize the model with the given configuration.
    
    Args:
        cfg (dict): Configuration dictionary containing model parameters
        
    Returns:
        model (nn.Module): Initialized model on appropriate device
    """
    print('hvt model init.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize body based on model type
    if cfg["model"].startswith("dino"):
        try:
            body = torch.hub.load("facebookresearch/dino:main", cfg["model"])
        except Exception as e:
            print(f"Error loading DINO model: {e}")
            raise
    else:
        try:
            body = timm.create_model(cfg["model"], pretrained=True)
        except Exception as e:
            print(f"Error loading timm model: {e}")
            raise
    
    # Initialize last layer based on hyperbolic configuration
    if cfg.get("hyp_c", 0) > 0:
        last = hypnn.ToPoincare(
            c=cfg["hyp_c"],
            ball_dim=cfg.get("emb", 128),
            riemannian=False,
            clip_r=cfg.get("clipr", None),
        )
    else:
        last = NormLayer()
    
    # Define model dimensions for different architectures
    model_dims = {
        "resnet50": 2048,
        "vit_base": 768,
        "vit_small": 384,
        "default": 384
    }
    
    # Get backbone dimension
    bdim = model_dims.get(cfg["model"], model_dims["default"])
    
    # Initialize head
    head = nn.Sequential(nn.Linear(bdim, cfg.get("emb", 128)), last)
    nn.init.constant_(head[0].bias.data, 0)
    nn.init.orthogonal_(head[0].weight.data)
    
    # Remove original head and freeze if specified
    rm_head(body)
    if cfg.get("freeze", None) is not None:
        freeze(body, cfg["freeze"])
    
    # Create and return model
    model = HeadSwitch(body, head)
    model = model.to(device)
    
    print(f"Model initialized on device: {device}")
    return model


class HeadSwitch(nn.Module):
    """
    A module that can switch between using a head and just normalizing the output.
    """
    def __init__(self, body, head):
        """
        Initialize HeadSwitch.
        
        Args:
            body (nn.Module): The backbone network
            head (nn.Module): The head network
        """
        super().__init__()
        self.body = body
        self.head = head
        self.norm = NormLayer()
    
    def forward(self, x, skip_head=False):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            skip_head (bool): Whether to skip the head and just normalize
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.body(x)
        if isinstance(x, tuple):
            x = x[0]
        
        if not skip_head:
            x = self.head(x)
        else:
            x = self.norm(x)
        return x


class NormLayer(nn.Module):
    """
    A layer that normalizes its input.
    """
    def forward(self, x):
        """
        Normalize input tensor along dimension 1.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        return F.normalize(x, p=2, dim=1)


def freeze(model, num_block):
    """
    Freeze specific parts of the model.
    
    Args:
        model (nn.Module): The model to freeze parts of
        num_block (int): Number of blocks to freeze
    """
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    try:
        fr(model.patch_embed)
        fr(model.pos_drop)
        for i in range(num_block):
            fr(model.blocks[i])
    except AttributeError as e:
        print(f"Warning: Could not freeze all requested layers. Error: {e}")


def rm_head(m):
    """
    Remove the head of the model by replacing it with Identity.
    
    Args:
        m (nn.Module): The model to remove the head from
    """
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())


# Utility function to check CUDA availability
def get_device():
    """
    Get the available device (GPU or CPU).
    
    Returns:
        torch.device: The available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
