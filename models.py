import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from protogcn_backbone import (
ProtoGCNBackbone ,
GCN_Block ,
Graph ,
full_body_graph_cfg ,
part_graph_cfg ,
load_pretrained_backbone ,
)


NTU_JOINT_GROUPS ={
"torso":[0 ,1 ,20 ,2 ,3 ],
"left_arm":[4 ,5 ,6 ,7 ,21 ,22 ],
"right_arm":[8 ,9 ,10 ,11 ,23 ,24 ],
"left_leg":[12 ,13 ,14 ,15 ],
"right_leg":[16 ,17 ,18 ,19 ],
}

PART_JOINTS ={
"torso":[0 ,1 ,20 ,2 ,3 ],
"left_arm":[4 ,5 ,6 ,7 ,21 ,22 ,20 ],
"right_arm":[8 ,9 ,10 ,11 ,23 ,24 ,20 ],
"left_leg":[12 ,13 ,14 ,15 ,0 ],
"right_leg":[16 ,17 ,18 ,19 ,0 ],
}

PART_EDGES ={
"torso":[(20 ,2 ),(2 ,3 ),(20 ,1 ),(1 ,0 )],
"left_arm":[(20 ,4 ),(4 ,5 ),(5 ,6 ),(6 ,7 ),(7 ,21 ),(7 ,22 )],
"right_arm":[(20 ,8 ),(8 ,9 ),(9 ,10 ),(10 ,11 ),(11 ,23 ),(11 ,24 )],
"left_leg":[(0 ,12 ),(12 ,13 ),(13 ,14 ),(14 ,15 )],
"right_leg":[(0 ,16 ),(16 ,17 ),(17 ,18 ),(18 ,19 )],
}

FULL_BODY_EDGES =[
(21 ,3 ),(3 ,4 ),
(21 ,9 ),(9 ,10 ),(10 ,11 ),(11 ,12 ),(12 ,24 ),(12 ,25 ),
(21 ,5 ),(5 ,6 ),(6 ,7 ),(7 ,8 ),(8 ,22 ),(8 ,23 ),
(21 ,2 ),(2 ,1 ),
(1 ,17 ),(17 ,18 ),(18 ,19 ),(19 ,20 ),
(1 ,13 ),(13 ,14 ),(14 ,15 ),(15 ,16 ),
]

BERT_DIM =768 
FEAT_DIM =384 
N_PARTS =len (NTU_JOINT_GROUPS )


NUM_PROTOTYPE =100 


#------------ Normalise adjacency matrix with symmetric degree normalisation
def _normalize (A ,device ):
    degree =torch .sum (A ,dim =1 )+0.001 
    D =torch .diag (degree )
    D_sqrt_inv =torch .sqrt (torch .inverse (D ))
    A_norm =torch .matmul (torch .matmul (D_sqrt_inv ,A ),D_sqrt_inv )
    return A_norm .to (device )


#------------ Return normalised full-body 25-joint adjacency matrix
def get_normalized_adj_fullbody (device ):
    num_joints =25 
    A =torch .zeros (num_joints ,num_joints ,dtype =torch .float )
    for (i ,j )in FULL_BODY_EDGES :
        A [i -1 ,j -1 ]=1 
        A [j -1 ,i -1 ]=1 
    return _normalize (A ,device )


#------------ Return normalised part-subgraph adjacency for the given body part
def get_normalized_adj_part (part ,device ):
    joints =PART_JOINTS [part ]
    raw_to_local ={raw :loc for loc ,raw in enumerate (joints )}
    num_joints =len (joints )
    A =torch .zeros (num_joints ,num_joints ,dtype =torch .float )
    for (i ,j )in PART_EDGES [part ]:
        li ,lj =raw_to_local [i ],raw_to_local [j ]
        A [li ,lj ]=1 
        A [lj ,li ]=1 
    return _normalize (A ,device )


#------------ Global average pool feature map over time, joints and bodies to (B,d)
def pool_feat_map (feat_map :torch .Tensor ,B :int ,M :int )->torch .Tensor :
    """(B*M, C, T, V) -> (B, C)  by mean over M, T, V."""
    C ,T ,V =feat_map .shape [1 ],feat_map .shape [2 ],feat_map .shape [3 ]
    f =feat_map .view (B ,M ,C ,T ,V )
    return f .mean (dim =(1 ,3 ,4 ))


#------------ Single ST-GCN decoder block with transposed temporal convolution
class ST_GCN_decode_unit (nn .Module ):
    #------------ Implementation of __init__
    def __init__ (self ,in_channels ,out_channels ,A_norm ,temporal_scale =1 ):
        super ().__init__ ()
        self .A_Norm =A_norm 
        self .M =nn .Parameter (torch .ones_like (A_norm ),requires_grad =True )
        self .temporal_scale =temporal_scale 

        self .spatial_conv =nn .Conv2d (in_channels ,out_channels ,kernel_size =(1 ,1 ))

        if temporal_scale >1 :
            self .temporal_conv =nn .ConvTranspose2d (
            out_channels ,out_channels ,
            kernel_size =(1 ,temporal_scale *2 ),
            stride =(1 ,temporal_scale ),
            padding =(0 ,temporal_scale //2 ),
            )
        else :
            self .temporal_conv =nn .Conv2d (
            out_channels ,out_channels ,
            kernel_size =(1 ,9 ),stride =(1 ,1 ),padding =(0 ,4 ),
            )

        self .res_layer =(
        nn .Conv2d (in_channels ,out_channels ,kernel_size =(1 ,1 ))
        if in_channels !=out_channels else nn .Identity ()
        )
        self .bn1 =nn .BatchNorm2d (out_channels )
        self .bn2 =nn .BatchNorm2d (out_channels )

    #------------ Implementation of forward
    def forward (self ,x ):
        res =self .res_layer (x )
        if self .temporal_scale >1 :
            res =F .interpolate (res ,scale_factor =(1 ,self .temporal_scale ),mode ='nearest')

        A =torch .mul (self .A_Norm ,self .M )
        A =torch .tile (A .unsqueeze (0 ),dims =(x .shape [1 ],1 ,1 ))

        x =x .permute (0 ,1 ,3 ,2 ).contiguous ()
        x =torch .einsum ('bctv,cvu->bctu',x ,A )
        x =x .permute (0 ,1 ,3 ,2 ).contiguous ()

        x =self .spatial_conv (x )
        x =F .relu (x ,inplace =True )
        x =self .bn1 (x )
        x =self .temporal_conv (x )
        x =F .relu (x ,inplace =True )
        x =self .bn2 (x )

        x =F .relu (x +res ,inplace =True )
        return x 


#------------ Construct nine-unit ST-GCN decoding stack restoring 3D joint coordinates
def _build_decoder (A_norm ):

    dec_in =[FEAT_DIM ,256 ,128 ,128 ,128 ,64 ,64 ,64 ,3 ]
    dec_out =[256 ,128 ,128 ,128 ,64 ,64 ,64 ,3 ,3 ]
    dec_scl =[1 ,2 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ]
    return nn .ModuleList ([
    ST_GCN_decode_unit (dec_in [i ],dec_out [i ],A_norm ,temporal_scale =dec_scl [i ])
    for i in range (9 )
    ])


#------------ MLP projection head mapping motion features to BERT embedding space
class BertProjectionHead (nn .Module ):

    #------------ Implementation of __init__
    def __init__ (self ,in_dim :int =FEAT_DIM ,out_dim :int =BERT_DIM ):
        super ().__init__ ()
        self .net =nn .Sequential (
        nn .Linear (in_dim ,512 ),
        nn .ReLU (inplace =True ),
        nn .LayerNorm (512 ),
        nn .Linear (512 ,out_dim ),
        )
        self .ln =nn .LayerNorm (out_dim )

    #------------ Implementation of forward
    def forward (self ,x :torch .Tensor )->torch .Tensor :
        proj =self .ln (self .net (x ))
        return F .normalize (proj ,dim =-1 )


#------------ Part Motion Autoencoder: ProtoGCN encoder-decoder for one body-part subgraph
class PartSTGCN (nn .Module ):


    #------------ Implementation of __init__
    def __init__ (self ,part ,device ):
        super ().__init__ ()
        self .part =part 
        self .num_owned =len (NTU_JOINT_GROUPS [part ])
        self .num_joints =len (PART_JOINTS [part ])


        A_norm =get_normalized_adj_part (part ,device )
        self .register_buffer ('A_norm',A_norm )


        self .encoder =ProtoGCNBackbone (
        graph_cfg =part_graph_cfg (PART_JOINTS [part ]),
        num_prototype =NUM_PROTOTYPE ,
        tcn_ms_cfg =[(3 ,1 ),(3 ,2 ),(3 ,3 ),(3 ,4 ),('max',3 ),'1x1'],
        )


        self .decoder =_build_decoder (A_norm )


        self .bert_proj =BertProjectionHead (in_dim =FEAT_DIM ,out_dim =BERT_DIM )


    #------------ Implementation of load_protogcn_pretrained
    def load_protogcn_pretrained (self ,ckpt_path ,verbose =True ):
        """Warm-start the part encoder from the pretrained ProtoGCN checkpoint.
        Only V-independent weights transfer (data_bn + per-block gcn.A are
        skipped because the part has fewer than 25 joints)."""
        return load_pretrained_backbone (self .encoder ,ckpt_path ,verbose =verbose )


    #------------ Implementation of _to_proto
    def _to_proto (self ,x ):
        """(B, C, F, V, M) -> (B, M, F, V, C) for the ProtoGCN backbone."""
        return x .permute (0 ,4 ,2 ,3 ,1 ).contiguous ()

    #------------ Implementation of encode
    def encode (self ,x ):
        """x: (B, C, F, V_part, M)  ->  pooled (B, 384)."""
        B ,C ,Fr ,V ,M =x .shape 
        feat =self .encoder (self ._to_proto (x ))
        return pool_feat_map (feat ,B ,M )

    #------------ Implementation of forward
    def forward (self ,x ):
        """
        x : (B, C, F, V_part, M)
        Returns (recon (B, 3, T_dec, V_part, M), bert_proj_feat (B, 768)).
        """
        B ,C ,Fr ,V ,M =x .shape 
        feat =self .encoder (self ._to_proto (x ))

        pooled =pool_feat_map (feat ,B ,M )
        bert_feat =self .bert_proj (pooled )


        d =feat .permute (0 ,1 ,3 ,2 ).contiguous ()
        for dec_layer in self .decoder :
            d =dec_layer (d )

        recon =d .view (B ,M ,3 ,V ,d .shape [3 ])
        recon =recon .permute (0 ,2 ,4 ,3 ,1 ).contiguous ()
        return recon ,bert_feat 

    #------------ Implementation of forward_with_task_feats
    def forward_with_task_feats (self ,x ):
        """Return reconstruction, frozen BERT projection, and z_task.

        This is used by the global AE training loop when we compute an
        auxiliary CE loss on the shared task feature.
        """
        feat_map_task ,z_task ,z_frozen_pooled =self ._encode_spatial (x )
        z_frozen_proj =self .bert_proj_frozen (z_frozen_pooled )

        d =feat_map_task .permute (0 ,1 ,3 ,2 ).contiguous ()
        for dec_layer in self .decoder :
            d =dec_layer (d )

        T_dec =d .shape [3 ]
        recon =d .view (x .shape [0 ],x .shape [4 ],3 ,x .shape [3 ],T_dec )
        recon =recon .permute (0 ,2 ,4 ,3 ,1 ).contiguous ()
        return recon ,z_frozen_proj ,z_task 


#------------ Continual Motion Autoencoder: full-body encoder with frozen SM-Enc and trainable extra layers
class GlobalSTGCNAE (nn .Module ):


    #------------ Implementation of __init__
    def __init__ (self ,part_models ,device ):
        super ().__init__ ()
        self .joint_groups =NTU_JOINT_GROUPS 
        self .device =device 


        self .part_models =nn .ModuleDict (part_models )
        for m in self .part_models .values ():
            for p in m .parameters ():
                p .requires_grad =False 


        A_norm =get_normalized_adj_fullbody (device )
        self .register_buffer ('A_norm',A_norm )


        self .global_encoder =ProtoGCNBackbone (
        graph_cfg =full_body_graph_cfg (),
        num_prototype =NUM_PROTOTYPE ,
        tcn_ms_cfg =[(3 ,1 ),(3 ,2 ),(3 ,3 ),(3 ,4 ),('max',3 ),'1x1'],
        )


        A_seed =torch .tensor (Graph (**full_body_graph_cfg ()).A ,dtype =torch .float32 )
        TCN =[(3 ,1 ),(3 ,2 ),(3 ,3 ),(3 ,4 ),('max',3 ),'1x1']
        self .extra_layers =nn .ModuleList ([
        GCN_Block (FEAT_DIM ,FEAT_DIM ,A_seed .clone (),stride =1 ,tcn_ms_cfg =TCN )
        for _ in range (4 )
        ])


        self .bert_proj_frozen =BertProjectionHead (in_dim =N_PARTS *FEAT_DIM ,
        out_dim =BERT_DIM )


        self .decoder =_build_decoder (A_norm )


    #------------ Implementation of train
    def train (self ,mode :bool =True ):
        """Keep the frozen pretrained streams in eval mode so their BatchNorm
        layers use the pretrained/finetuned running statistics and never update
        them. Only the trainable extra_layers / decoder / bert_proj_frozen
        follow the requested train/eval mode."""
        super ().train (mode )

        for m in self .part_models .values ():
            m .eval ()
        return self 


    #------------ Implementation of load_protogcn_pretrained
    def load_protogcn_pretrained (self ,ckpt_path ,verbose =True ):

        return load_pretrained_backbone (self .global_encoder ,ckpt_path ,verbose =verbose )


    #------------ Implementation of _to_proto
    def _to_proto (self ,x ):

        return x .permute (0 ,4 ,2 ,3 ,1 ).contiguous ()

    #------------ Implementation of _part_semantic
    def _part_semantic (self ,x ):

        feats =[]
        for part in self .joint_groups :
            xp =x [:,:,:,PART_JOINTS [part ],:]
            feats .append (self .part_models [part ].encode (xp ))
        return torch .cat (feats ,dim =1 )

    #------------ Implementation of _encode_spatial
    def _encode_spatial (self ,x ):

        B ,C ,Fr ,V ,M =x .shape 


        z_frozen_pooled =self ._part_semantic (x )


        feat =self .global_encoder (self ._to_proto (x ))


        for blk in self .extra_layers :
            feat ,_ =blk (feat )
        feat_map_task =feat 

        z_task =pool_feat_map (feat_map_task ,B ,M )
        return feat_map_task ,z_task ,z_frozen_pooled 

    #------------ Implementation of encode_ztask_only
    def encode_ztask_only (self ,x :torch .Tensor )->torch .Tensor :
        """z_task (B, 384) only — used by the copy/slow model in the blended
        classifier phase (skips the BERT projection)."""
        _ ,z_task ,_ =self ._encode_spatial (x )
        return z_task 

    #------------ Implementation of encode
    def encode (self ,x ):
        """Returns (z_task (B,384), z_frozen_proj (B,768))."""
        _ ,z_task ,z_frozen_pooled =self ._encode_spatial (x )
        z_frozen_proj =self .bert_proj_frozen (z_frozen_pooled )
        return z_task ,z_frozen_proj 

    #------------ Implementation of forward
    def forward (self ,x ):
        """Full AE forward. Returns (recon (B,3,F,25,M), z_frozen_proj (B,768))."""
        B ,C ,Fr ,V ,M =x .shape 
        feat_map_task ,z_task ,z_frozen_pooled =self ._encode_spatial (x )
        z_frozen_proj =self .bert_proj_frozen (z_frozen_pooled )


        d =feat_map_task .permute (0 ,1 ,3 ,2 ).contiguous ()
        for dec_layer in self .decoder :
            d =dec_layer (d )

        T_dec =d .shape [3 ]
        recon =d .view (B ,M ,3 ,V ,T_dec )
        recon =recon .permute (0 ,2 ,4 ,3 ,1 ).contiguous ()
        return recon ,z_frozen_proj 

    #------------ Implementation of forward_with_task_feats
    def forward_with_task_feats (self ,x ):
        """Return (recon, z_frozen_proj, z_task).

        Mirrors `forward` but also returns the pooled `z_task` used by CE head.
        """
        B ,C ,Fr ,V ,M =x .shape 
        feat_map_task ,z_task ,z_frozen_pooled =self ._encode_spatial (x )
        z_frozen_proj =self .bert_proj_frozen (z_frozen_pooled )

        d =feat_map_task .permute (0 ,1 ,3 ,2 ).contiguous ()
        for dec_layer in self .decoder :
            d =dec_layer (d )

        T_dec =d .shape [3 ]
        recon =d .view (B ,M ,3 ,V ,T_dec )
        recon =recon .permute (0 ,2 ,4 ,3 ,1 ).contiguous ()
        return recon ,z_frozen_proj ,z_task 


#------------ Auxiliary per-task cross-entropy head applied to the CM-AE bottleneck
class CE_Classifier (nn .Module ):
    """Plain linear probe on z_task (384-d), trained with cross-entropy only.
    Name retained for backward compatibility with cmain.py / checkpoints."""

    #------------ Implementation of __init__
    def __init__ (self ,num_classes :int ,device :str ):
        super ().__init__ ()
        self .device =device 
        self .num_classes =num_classes 
        self .linear =nn .Linear (FEAT_DIM ,num_classes )
        self .to (device )

    #------------ Implementation of forward
    def forward (self ,z_task :torch .Tensor )->torch .Tensor :
        return self .linear (z_task )

    #------------ Implementation of compute_loss
    def compute_loss (self ,z_task :torch .Tensor ,labels :torch .Tensor ):
        logits =self .forward (z_task )
        ce_loss =F .cross_entropy (logits ,labels )
        return ce_loss ,{"ce_loss":ce_loss .item ()}


#------------ Linear CIL probe trained on frozen z_task features for all accumulated classes
class LinearCILClassifier (nn .Module ):
    """Linear probe: 384 -> num_cil_classes.  Uses z_task only."""
    #------------ Implementation of __init__
    def __init__ (self ,num_classes ,device ):
        super ().__init__ ()
        self .fc =nn .Linear (FEAT_DIM ,num_classes )
        self .to (device )

    #------------ Implementation of forward
    def forward (self ,feats ):
        return self .fc (feats )
