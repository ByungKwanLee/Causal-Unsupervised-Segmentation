import torch
import torch.nn as nn
from modules.segment_module import HeadSegment, Decoder, ProjectionSegment

class Segment_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        ##################################################################################
        # dropout
        self.dropout = torch.nn.Dropout(p=0.1)
        ##################################################################################

        ##################################################################################
        # MLP Head
        self.head = HeadSegment(args.dim, args.reduced_dim)
        self.projection_head = ProjectionSegment(nn.Conv2d(args.reduced_dim, args.projection_dim, kernel_size=1), is_untrans=True)
        ##################################################################################

        ##################################################################################
        # MLP EMA Head
        self.head_ema = HeadSegment(args.dim, args.reduced_dim)
        self.projection_head_ema = ProjectionSegment(nn.Conv2d(args.reduced_dim, args.projection_dim, kernel_size=1), is_untrans=True)
        self.linear = ProjectionSegment(nn.Conv2d(args.reduced_dim, args.n_classes, kernel_size=1), is_untrans=False)
        ##################################################################################

class Segment_TR(nn.Module):
    def __init__(self, args):
        super().__init__()

        ##################################################################################
        # dropout
        self.dropout = nn.Dropout(p=0.1)
        ##################################################################################
        
        ##################################################################################
        # TR Decoder Head 
        self.head = Decoder(args)
        self.projection_head = ProjectionSegment(nn.Conv2d(args.reduced_dim, args.projection_dim, kernel_size=1), is_untrans=True)
        ##################################################################################

        ##################################################################################
        # TR Decoder EMA Head
        self.head_ema = Decoder(args)
        self.projection_head_ema = ProjectionSegment(nn.Conv2d(args.reduced_dim, args.projection_dim, kernel_size=1), is_untrans=True)
        self.linear = ProjectionSegment(nn.Conv2d(args.reduced_dim, args.n_classes, kernel_size=1), is_untrans=False)
        ##################################################################################


    
