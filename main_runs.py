import os 
import sys 


os .environ .setdefault ("TRANSFORMERS_VERBOSITY","error")
os .environ .setdefault ("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")

import argparse 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from torch .optim import Adam 
from torch .optim .lr_scheduler import ReduceLROnPlateau 
from torch .utils .data import DataLoader ,TensorDataset 

from models import (
PartSTGCN ,
GlobalSTGCNAE ,
CE_Classifier ,
LinearCILClassifier ,
NTU_JOINT_GROUPS ,
PART_JOINTS ,
)
from dataloader import (
TASK_SPLITS ,
configure_protocol ,
VALID_PROTOCOLS ,
PKU_NUM_CLASSES ,
create_part_loader_all_classes ,
create_part_loader_pku ,
build_task_data ,
NTUDataset ,
resize_sequence_to_64 ,
)
from bert_feat import BertFeatureBank ,SentenceBertBank ,DATASET_CLASS_NAMES 


DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"


DATASET ="ntu60"
PROTOCOL ="xview"

DEFAULT_NPY_PATH ={
"ntu60":"data/ntu/npy60/",
"ntu120":"data/ntu/npy120/",
}
DEFAULT_BERT_FEAT_PATH ={
"ntu60":"data/ntu/bert_feats_60.npy",
"ntu120":"data/ntu/bert_feats_120.npy",
}
DEFAULT_SENT_FEAT_PATH ={
"ntu60":"data/ntu/bert_sentence_feats_60.npy",
"ntu120":"data/ntu/bert_sentence_feats_120.npy",
}

BERT_DATASET_KEY ={"ntu60":"ntu","ntu120":"ntu120"}

NPY_PATH =DEFAULT_NPY_PATH [DATASET ]

BATCH_SIZE =32 
PART_EPOCHS =100 
GLOBAL_EPOCHS =50 
CIL_CLF_EPOCHS =100 

LR_PART =1e-3 
LR_GLOBAL =1e-4 
LR_CIL =1e-4 
LR_SCHEDULE_FACTOR =0.8 
LR_SCHEDULE_PATIENCE =8 
CE_LOSS_WEIGHT =0.3 
CE_CLASS_MODE =1 


MAX_TASK =-1 
START_TASK =1 
CHECKPOINT_DIR ="ntu60_xview"
print (DEVICE )


PROTOGCN_J_CKPT =None 
PART_LOAD_PROTOGCN =False 

GLOBAL_MODE ="autoencoder"


PART_SOURCE ="pku"
PKU_NPY_PATH ="data/pku/npy"

S_SAMPLES_PER_CLASS =None 
DATA_FRACTION =1.0 


BERT_MODEL_PATH ="./bert-base-uncased"
NTU_BERT_FEAT_PATH =DEFAULT_BERT_FEAT_PATH [DATASET ]
PKU_BERT_FEAT_PATH ="data/pku/bert_feats.npy"
NTU_SENT_FEAT_PATH =DEFAULT_SENT_FEAT_PATH [DATASET ]


LAMBDA_ALIGN =0.3 


NOVELTY_THRESHOLD =0.45 


NOVEL_SAMPLE_FRACTION =0.5 
ALPHA_INIT =0.5 


TRAIN_PARTS =False 
TRAIN_GLOBAL =True 
TRAIN_CLASSIFIER =True 


GLOBAL_SEED =-1 


SHUFFLE_VARIANT =-1 


#------------ Parse all command-line arguments and return the argument Namespace
def _parse_args ():
    parser =argparse .ArgumentParser (
    description ="CIL skeleton action recognition (PartSTGCN + GlobalSTGCNAE + SA-CIL)."
    )
    parser .add_argument ("--dataset",type =str ,default =DATASET ,
    choices =["ntu60","ntu120"],
    help ="Evaluation dataset (class set + task splits).")
    parser .add_argument ("--protocol",type =str ,default =PROTOCOL ,
    choices =["xsub","xview","xset"],
    help ="ntu60: xsub|xview   ntu120: xsub|xset")
    parser .add_argument ("--part_ckpt_dir",type =str ,default =CHECKPOINT_DIR ,
    help ="Directory holding the frozen part_*.pt models. "
    "Defaults to --checkpoint_dir. Set this to reuse the "
    "same PKU-pretrained parts across protocols.")
    parser .add_argument ("--npy_path",type =str ,default =None )
    parser .add_argument ("--pku_npy_path",type =str ,default =PKU_NPY_PATH )
    parser .add_argument ("--checkpoint_dir",type =str ,default =CHECKPOINT_DIR )
    parser .add_argument ("--bert_model_path",type =str ,default =BERT_MODEL_PATH )
    parser .add_argument ("--ntu_bert_feat_path",type =str ,default =None )
    parser .add_argument ("--pku_bert_feat_path",type =str ,default =PKU_BERT_FEAT_PATH )
    parser .add_argument ("--ntu_sent_feat_path",type =str ,default =None )
    parser .add_argument ("--batch_size",type =int ,default =BATCH_SIZE )
    parser .add_argument ("--part_epochs",type =int ,default =PART_EPOCHS )
    parser .add_argument ("--global_epochs",type =int ,default =GLOBAL_EPOCHS )
    parser .add_argument ("--cil_clf_epochs",type =int ,default =CIL_CLF_EPOCHS )
    parser .add_argument ("--lr_part",type =float ,default =LR_PART )
    parser .add_argument ("--lr_global",type =float ,default =LR_GLOBAL )
    parser .add_argument ("--lr_cil",type =float ,default =LR_CIL )
    parser .add_argument ("--lr_schedule_factor",type =float ,default =LR_SCHEDULE_FACTOR )
    parser .add_argument ("--lr_schedule_patience",type =int ,default =LR_SCHEDULE_PATIENCE )
    parser .add_argument ("--max_task",type =int ,default =MAX_TASK )
    parser .add_argument ("--start_task",type =int ,default =START_TASK )
    parser .add_argument ("--part_source",type =str ,default =PART_SOURCE ,
    choices =["ntu","pku"])
    parser .add_argument ("--s_samples_per_class",type =int ,default =-1 ,
    help ="Max samples per class for part pretraining. "
    "-1 or 0 = use ALL available samples (no cap).")
    parser .add_argument ("--data_fraction",type =float ,default =DATA_FRACTION )
    parser .add_argument ("--lambda_align",type =float ,default =LAMBDA_ALIGN )
    parser .add_argument ("--novelty_threshold",type =float ,default =NOVELTY_THRESHOLD )
    parser .add_argument ("--novel_sample_fraction",type =float ,default =NOVEL_SAMPLE_FRACTION ,
    help ="Fraction of each novel class's samples to add into "
    "the current task's global training set (1.0=all).")
    parser .add_argument ("--ce_loss_weight",type =float ,default =CE_LOSS_WEIGHT ,
    help ="Weight of the CE loss term added to global AE training.")
    parser .add_argument ("--ce_class_mode",type =int ,default =CE_CLASS_MODE ,
    choices =[1 ,2 ,3 ],
    help ="CE class set: 1=current+novel, 2=novel only, 3=current only.")
    parser .add_argument ("--alpha_init",type =float ,default =ALPHA_INIT )
    parser .add_argument ("--seed",type =int ,default =GLOBAL_SEED ,
    help ="Random seed for Phase 2 (global AE) and Phase 3 "
    "(classifier) training. 0/1/2 are the standard values. "
    "-1 disables seeding (original behaviour). "
    "Phase 1 (part pretraining) is NEVER seeded by this flag.")
    parser .add_argument ("--shuffle_variant",type =int ,default =SHUFFLE_VARIANT ,
    choices =[-1 ,0 ,1 ,2 ],
    help ="Class-order shuffle variant applied to TASK_SPLITS "
    "before Phase 2 and Phase 3. "
    "0/1/2 = three reproducible shuffles (seeded at "
    "100/101/102, independent of --seed). "
    "-1 = keep the original fixed class order.")
    _b =lambda x :x .lower ()=='true'
    parser .add_argument ("--train_parts",type =_b ,default =TRAIN_PARTS ,
    help ="Train part models (True/False).")
    parser .add_argument ("--train_global",type =_b ,default =TRAIN_GLOBAL ,
    help ="Train global autoencoder (True/False).")
    parser .add_argument ("--train_classifier",type =_b ,default =TRAIN_CLASSIFIER ,
    help ="Train and evaluate classifiers (True/False).")
    return parser .parse_args ()


_args =_parse_args ()


DATASET =_args .dataset 
PROTOCOL =_args .protocol 
assert PROTOCOL in VALID_PROTOCOLS [DATASET ],f"protocol '{PROTOCOL }' invalid for {DATASET }; use {VALID_PROTOCOLS [DATASET ]}"
_N_TASKS =configure_protocol (DATASET ,PROTOCOL )
BERT_KEY =BERT_DATASET_KEY [DATASET ]

NPY_PATH =_args .npy_path or DEFAULT_NPY_PATH [DATASET ]
PKU_NPY_PATH =_args .pku_npy_path 
CHECKPOINT_DIR =_args .checkpoint_dir 
PART_CKPT_DIR =_args .part_ckpt_dir or CHECKPOINT_DIR 
BERT_MODEL_PATH =_args .bert_model_path 
NTU_BERT_FEAT_PATH =_args .ntu_bert_feat_path or DEFAULT_BERT_FEAT_PATH [DATASET ]
PKU_BERT_FEAT_PATH =_args .pku_bert_feat_path 
NTU_SENT_FEAT_PATH =_args .ntu_sent_feat_path or DEFAULT_SENT_FEAT_PATH [DATASET ]
BATCH_SIZE =_args .batch_size 
PART_EPOCHS =_args .part_epochs 
GLOBAL_EPOCHS =_args .global_epochs 
CIL_CLF_EPOCHS =_args .cil_clf_epochs 
LR_PART =_args .lr_part 
LR_GLOBAL =_args .lr_global 
LR_CIL =_args .lr_cil 
LR_SCHEDULE_FACTOR =_args .lr_schedule_factor 
LR_SCHEDULE_PATIENCE =_args .lr_schedule_patience 
MAX_TASK =_N_TASKS if _args .max_task in (-1 ,0 )else _args .max_task 
START_TASK =_args .start_task 
PART_SOURCE =_args .part_source 
S_SAMPLES_PER_CLASS =None if _args .s_samples_per_class in (-1 ,0 )else _args .s_samples_per_class 
DATA_FRACTION =_args .data_fraction 
LAMBDA_ALIGN =_args .lambda_align 
NOVELTY_THRESHOLD =_args .novelty_threshold 
NOVEL_SAMPLE_FRACTION =_args .novel_sample_fraction 
CE_LOSS_WEIGHT =_args .ce_loss_weight 
CE_CLASS_MODE =_args .ce_class_mode 
TRAIN_GLOBAL =_args .train_global 
TRAIN_CLASSIFIER =_args .train_classifier 
GLOBAL_SEED =_args .seed 
SHUFFLE_VARIANT =_args .shuffle_variant 

os .makedirs (CHECKPOINT_DIR ,exist_ok =True )
os .makedirs (PART_CKPT_DIR ,exist_ok =True )


#------------ Implementation of apply_shuffle_variant
def apply_shuffle_variant (variant :int )->None :
    """
    Shuffle the class order in TASK_SPLITS for Phase 2 and Phase 3.

    Collects all class IDs from the current TASK_SPLITS, randomly permutes
    them using a deterministic per-variant seed (100 + variant), and
    re-partitions them into tasks of the same size as the original splits.
    The shuffled ordering is written back into TASK_SPLITS in place.

    This function modifies the module-level TASK_SPLITS dict (imported from
    dataloader_r) so that build_task_data, build_global_class_map, and all
    loader helpers automatically see the new class order.

    Args:
        variant : 0, 1, or 2. Each value produces a different but fixed
                  shuffle via seeds 100, 101, 102 respectively.
    """
    import random 
    variant_seed =100 +variant 
    task_size =len (TASK_SPLITS [1 ])
    all_classes =[c for t in sorted (TASK_SPLITS )for c in TASK_SPLITS [t ]]

    rng =random .Random (variant_seed )
    rng .shuffle (all_classes )

    for i ,t in enumerate (sorted (TASK_SPLITS )):
        TASK_SPLITS [t ]=all_classes [i *task_size :(i +1 )*task_size ]

    print (f"  [Shuffle] variant={variant } (seed={variant_seed })  "
    f"new class order per task:")
    for t in sorted (TASK_SPLITS ):
        print (f"    task {t }: {TASK_SPLITS [t ]}")


if SHUFFLE_VARIANT >=0 :
    apply_shuffle_variant (SHUFFLE_VARIANT )
    print (f"  [Shuffle] TASK_SPLITS updated for variant {SHUFFLE_VARIANT }")
else :
    print (f"  [Shuffle] variant=-1 → using original fixed TASK_SPLITS")


#------------ Set random seeds for torch, numpy and python random for reproducibility
def set_global_seed (seed :int )->None :
    """
    Set random seeds for Phase 2 and Phase 3 training.

    Affects: torch, torch.cuda, numpy, and Python random.
    Phase 1 (part pretraining) is intentionally NOT seeded by this function
    so that part models remain reproducible under their own default init.

    Args:
        seed : integer seed value (0, 1, or 2 in standard usage).
               If seed < 0 this function is a no-op.
    """
    if seed <0 :
        return 
    import random ,numpy as np 
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    if torch .cuda .is_available ():
        torch .cuda .manual_seed (seed )
        torch .cuda .manual_seed_all (seed )
    print (f"  [Seed] global seed set to {seed } (Phase 2 / Phase 3 only)")


#------------ Return checkpoint path for the given body-part PM-AE
def part_ckpt_path (part ):


    return os .path .join (PART_CKPT_DIR ,f"part_{part }.pt")

#------------ Return checkpoint path for the CM-AE at the given task
def ae_global_ckpt_path (task ):
    return os .path .join (CHECKPOINT_DIR ,f"global_ae_task{task }.pt")

#------------ Return checkpoint path for the linear CIL classifier at the given task
def cil_clf_ckpt_path (task ):
    return os .path .join (CHECKPOINT_DIR ,f"cil_classifier_task{task }.pt")

#------------ Save PM-AE state dict to the part checkpoint directory
def save_part_model (model ,part ):
    torch .save (model .state_dict (),part_ckpt_path (part ))
    print (f"    [saved] {part_ckpt_path (part )}")

#------------ Load PM-AE state dict from the part checkpoint directory
def load_part_model (part ,device ):
    model =PartSTGCN (part ,device ).to (device )
    model .load_state_dict (torch .load (part_ckpt_path (part ),map_location =device ))
    model .eval ()
    return model 

#------------ Save CM-AE state dict to the checkpoint directory for the given task
def save_ae_global_model (model ,task ):
    """
    Save the full GlobalSTGCNAE state dict.
    Previously also saved a separate global_ae_enc_task{N}.pt containing only
    extra_layers — that is no longer needed since the next task's fast model is
    initialised by loading this full checkpoint directly, carrying the decoder,
    bert_proj_frozen, and extra_layers all together.
    """
    torch .save (model .state_dict (),ae_global_ckpt_path (task ))
    print (f"    [saved] {ae_global_ckpt_path (task )}")

#------------ Save CIL classifier state dict for the given task
def save_cil_classifier (model ,task ):
    torch .save (model .state_dict (),cil_clf_ckpt_path (task ))
    print (f"    [saved] {cil_clf_ckpt_path (task )}")

#------------ Load CM-AE and reinitialise from the previous task checkpoint if available
def load_ae_global_model (task ,part_models ,device ):
    """
    Load a saved GlobalSTGCNAE from global_ae_task{task}.pt.
    Returns the model in eval() mode.  Call .train() and rebuild the
    optimizer after loading if further training is needed.
    The requires_grad flags are set by GlobalSTGCNAE.__init__:
        global_layers  → False (frozen backbone)
        part_models    → False (frozen part encoders)
        extra_layers   → True  (trainable)
        decoder        → True  (trainable)
        bert_proj_frozen → True (trainable)
    so the returned model is immediately ready for use as a training starting
    point with no manual requires_grad adjustments.
    """
    model =GlobalSTGCNAE (
    part_models ={p :part_models [p ]for p in NTU_JOINT_GROUPS },
    device =device ,
    ).to (device )
    model .load_state_dict (torch .load (ae_global_ckpt_path (task ),map_location =device ))
    model .eval ()
    return model 


#------------ Build the cumulative class-to-index mapping across all tasks
def build_global_class_map ():
    mapping ,idx ={},0 
    for t in range (1 ,MAX_TASK +1 ):
        for a in TASK_SPLITS [t ]:
            if a not in mapping :
                mapping [a ]=idx 
                idx +=1 
    return mapping 


#------------ DataLoader for current task samples only
def build_current_task_loader (npy_path ,task_id ,global_map ,batch_size ,
shuffle =True ,data_fraction =1.0 ):
    Xtr ,Ytr ,_ ,_ =build_task_data (npy_path ,task_id ,data_fraction =data_fraction )
    Y_mapped =[global_map [a ]for a in Ytr ]
    ds =NTUDataset (Xtr ,Y_mapped ,Ytr )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle )

#------------ DataLoader combining current task and selectively replayed previous samples
def build_task_plus_novel_loader (npy_path ,task_id ,novel_ids ,global_map ,
batch_size ,novel_fraction =1.0 ,data_fraction =1.0 ):
    """[ADD] Training set for the global AE at task `task_id`:
       ALL samples of the current task's classes, PLUS `novel_fraction` of the
       samples of each novel class (novel_ids are previous-task classes judged
       previous classes uncovered by the current task, re-shown for
       rehearsal)."""

    Xtr ,Ytr ,_ ,_ =build_task_data (npy_path ,task_id ,data_fraction =data_fraction )
    X =list (Xtr );Y =[global_map [a ]for a in Ytr ];R =list (Ytr )

    added =0 
    if novel_ids and novel_fraction >0.0 :


        want =set (novel_ids )
        owning_tasks =sorted ({t for t in range (1 ,task_id )
        if want &set (TASK_SPLITS [t ])})
        for t in owning_tasks :
            Xt ,Yt ,_ ,_ =build_task_data (npy_path ,t ,data_fraction =data_fraction )
            Xn =[x for x ,y in zip (Xt ,Yt )if y in want ]
            Yn =[y for y in Yt if y in want ]
            Xn ,Yn =_subsample_frac (Xn ,Yn ,novel_fraction )
            X .extend (Xn );Y .extend ([global_map [a ]for a in Yn ]);R .extend (Yn )
            added +=len (Xn )

    print (f"  [GlobalData] task {task_id }: {len (Xtr )} current + {added } rehearsed "
    f"(fraction={novel_fraction :.2f}) = {len (X )} samples")
    ds =NTUDataset (X ,Y ,R )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =True )


#------------ Subsample a fixed fraction of samples from tensors X and y
def _subsample_frac (X ,y ,fraction ,seed =42 ):
    """[ADD] keep `fraction` of samples per class (>=1 each). 1.0 -> unchanged."""
    if fraction >=1.0 :
        return X ,y 
    import random 
    from collections import defaultdict as _dd 
    rng =random .Random (seed );buckets =_dd (list )
    for i ,lab in enumerate (y ):
        buckets [lab ].append (i )
    keep =[]
    for lab in sorted (buckets ):
        idxs =buckets [lab ]
        k =max (1 ,int (round (len (idxs )*fraction )))
        keep .extend (rng .sample (idxs ,k ))
    keep .sort ()
    return [X [i ]for i in keep ],[y [i ]for i in keep ]


#------------ DataLoader over all classes seen up to the given task for classifier training
def build_cil_train_loader (npy_path ,upto_task ,global_map ,batch_size ,
data_fraction =1.0 ):
    X ,Y ,R =[],[],[]
    for t in range (1 ,upto_task +1 ):
        Xtr ,Ytr ,_ ,_ =build_task_data (npy_path ,t ,data_fraction =data_fraction )
        X .extend (Xtr )
        Y .extend ([global_map [a ]for a in Ytr ])
        R .extend (Ytr )
    ds =NTUDataset (X ,Y ,R )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =True )

#------------ DataLoader over all test samples seen up to the given task for evaluation
def build_cil_test_loader (npy_path ,upto_task ,global_map ,batch_size ,
data_fraction =1.0 ):
    X ,Y ,R =[],[],[]
    for t in range (1 ,upto_task +1 ):
        _ ,_ ,Xte ,Yte =build_task_data (npy_path ,t ,data_fraction =data_fraction )
        X .extend (Xte )
        Y .extend ([global_map [a ]for a in Yte ])
        R .extend (Yte )
    ds =NTUDataset (X ,Y ,R )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =False )


#------------ Run one epoch of PM-AE training with reconstruction and semantic alignment losses
def train_part (model ,loader ,optimizer ,scheduler ,bert_bank ):
    model .train ()
    for epoch in range (PART_EPOCHS ):
        epoch_loss ,num_batches =0.0 ,0 
        for x ,_ ,raw_ids in loader :
            x =x .to (DEVICE )
            raw_ids =raw_ids .to (DEVICE )
            recon ,bert_proj_feat =model (x )
            recon_loss =F .mse_loss (recon ,x )
            z_bert =bert_bank .get_batch (raw_ids )
            z_bert =F .normalize (z_bert ,dim =-1 )
            align_loss =F .mse_loss (bert_proj_feat ,z_bert )
            loss =recon_loss +LAMBDA_ALIGN *align_loss 
            optimizer .zero_grad ()
            loss .backward ()
            optimizer .step ()
            epoch_loss +=loss .item ()
            num_batches +=1 
        avg =epoch_loss /max (num_batches ,1 )
        lr =optimizer .param_groups [0 ]["lr"]
        pct =100 *(epoch +1 )/PART_EPOCHS 
        print (f"    [Part-{model .part }] {pct :5.1f}%  ep {epoch +1 }/{PART_EPOCHS }  "
        f"loss={avg :.6f}  lr={lr :.2g}")
        scheduler .step (avg )


#------------ Run one epoch of CM-AE training with reconstruction, alignment and CE losses
def train_global_ae (model ,loader ,optimizer ,scheduler ,ntu_bert_bank ,
task_id ,ce_head ,ce_label_map ,ce_loss_weight ,
num_epochs =None ):
    """
    Train a GlobalSTGCNAE for one task.

    Loss = MSE(recon, x)                                     — reconstruction
         + LAMBDA_ALIGN   * MSE(z_frozen_proj, bert_anchor)  — BERT alignment
         + CE_WEIGHT * CE(logits, y_subset)                  — auxiliary CE on z_task
    """
    epochs =num_epochs if num_epochs is not None else GLOBAL_EPOCHS 
    model .train ()
    if ce_head is not None :
        ce_head .train ()

    for epoch in range (epochs ):
        epoch_loss ,num_batches =0.0 ,0 
        for x ,labels ,raw_ids in loader :
            x =x .to (DEVICE )
            labels =labels .to (DEVICE )
            raw_ids =raw_ids .to (DEVICE )

            recon ,z_frozen_proj ,z_task =model .forward_with_task_feats (x )
            T_in ,T_out =x .shape [2 ],recon .shape [2 ]
            if T_out !=T_in :
                recon =recon [:,:,:T_in ,:,:]

            recon_loss =F .mse_loss (recon ,x )

            z_bert =ntu_bert_bank .get_batch (raw_ids )
            z_bert =F .normalize (z_bert ,dim =-1 )
            align_loss =F .mse_loss (z_frozen_proj ,z_bert )

            ce_loss =torch .tensor (0.0 ,device =DEVICE )
            if ce_head is not None and ce_loss_weight >0.0 :
                mask =torch .zeros (raw_ids .shape ,dtype =torch .bool ,device =DEVICE )
                mapped_labels =[]
                for i ,raw_id in enumerate (raw_ids ):
                    if raw_id .item ()in ce_label_map :
                        mask [i ]=True 
                        mapped_labels .append (ce_label_map [raw_id .item ()])
                if mask .any ():
                    logits =ce_head (z_task )
                    mapped_labels =torch .tensor (mapped_labels ,device =DEVICE ,dtype =torch .long )
                    ce_loss =F .cross_entropy (logits [mask ],mapped_labels )

            loss =recon_loss +LAMBDA_ALIGN *align_loss 
            if ce_loss_weight >0.0 :
                loss =loss +ce_loss_weight *ce_loss 

            optimizer .zero_grad ()
            loss .backward ()
            optimizer .step ()
            epoch_loss +=loss .item ()
            num_batches +=1 

        avg =epoch_loss /max (num_batches ,1 )
        lr =optimizer .param_groups [0 ]["lr"]
        pct =100 *(epoch +1 )/epochs 
        print (f"    [GlobalAE] {pct :5.1f}%  ep {epoch +1 }/{epochs }  "
        f"loss={avg :.6f}  [recon+{LAMBDA_ALIGN }*align+{ce_loss_weight }*ce]  lr={lr :.2g}")
        scheduler .step (avg )


#------------ Identify current-task classes that are semantically novel relative to all seen classes
def find_novel_classes (task ,sentence_bank ,threshold =NOVELTY_THRESHOLD ):
    current_ids =list (TASK_SPLITS [task ])
    previous_ids =[]
    for t in range (1 ,task ):
        previous_ids .extend (TASK_SPLITS [t ])

    novel_ids =sentence_bank .novel_classes (
    current_ids =current_ids ,
    previous_ids =previous_ids ,
    threshold =threshold ,
    )


    _names =DATASET_CLASS_NAMES .get (BERT_KEY ,{})
    #------------ Implementation of _nm
    def _nm (i ):
        return _names .get (i ,"?")

    print (f"  [Novelty] Task {task }: {len (novel_ids )}/{len (current_ids )} novel classes "
    f"(threshold={threshold :.2f})")

    for cid in current_ids :
        tag ="NOVEL"if cid in novel_ids else "known"


        if not previous_ids :
            print (f"    [{tag }] class {cid :3d} ({_nm (cid )})  — no previous classes")
            continue 


        sims =sentence_bank .cosine_similarity_matrix ([cid ],previous_ids )[0 ]
        k =min (3 ,len (previous_ids ))
        topv ,topi =sims .topk (k )
        matches =[(previous_ids [topi [j ].item ()],topv [j ].item ())for j in range (k )]
        best_f1 =matches [0 ][1 ]

        head =f"    [{tag }] class {cid :3d} ({_nm (cid )})  best_F1={best_f1 :.3f}"

        if tag =="known":

            reasons =[(pid ,f1 )for pid ,f1 in matches if f1 >=threshold ]or [matches [0 ]]
            reason_str =" ; ".join (f"{pid } ({_nm (pid )}) F1={f1 :.3f}"for pid ,f1 in reasons )
            print (head +f"  <= decided KNOWN by previous: {reason_str }")
        else :
            near_str =" ; ".join (f"{pid } ({_nm (pid )}) F1={f1 :.3f}"for pid ,f1 in matches )
            print (head +f"  (all < {threshold :.2f}; nearest previous: {near_str })")

    return novel_ids 


#------------ Find previous classes not semantically covered by the current task for selective replay
def find_uncovered_previous_classes (task ,sentence_bank ,threshold ,
already_rehearsed ):
    """For task t, scan every PREVIOUS class (tasks 1..t-1) and decide whether
    the CURRENT task covers it. A previous class is UNCOVERED (-> rehearse) if
    NO current-task class is similar to it (max SPICE F1 to the current classes
    < threshold). Classes already rehearsed by this mechanism in an earlier task
    are excluded from the comparison entirely.

    Returns: list of previous class ids to rehearse in this task.
    """
    current_ids =list (TASK_SPLITS [task ])
    previous_ids =[c for t in range (1 ,task )for c in TASK_SPLITS [t ]]

    candidates =[c for c in previous_ids if c not in already_rehearsed ]

    _names =DATASET_CLASS_NAMES .get (BERT_KEY ,{})
    #------------ Implementation of _nm
    def _nm (i ):
        return _names .get (i ,"?")

    if not candidates :
        print (f"  [Rehearsal] Task {task }: no candidate previous classes "
        f"(all {len (previous_ids )} already rehearsed or none).")
        return []

    uncovered =[]
    print (f"  [Rehearsal] Task {task }: checking {len (candidates )} previous "
    f"classes against {len (current_ids )} current (threshold={threshold :.2f})")
    for pid in candidates :

        sims =sentence_bank .cosine_similarity_matrix ([pid ],current_ids )[0 ]
        best =sims .max ().item ()
        j =int (sims .argmax ())
        best_cur =current_ids [j ]
        if best <threshold :
            uncovered .append (pid )
            print (f"    [UNCOVERED] prev {pid :3d} ({_nm (pid )})  best match to current "
            f"= {best_cur } ({_nm (best_cur )}) F1={best :.3f} < {threshold :.2f}  -> REHEARSE")
        else :
            print (f"    [covered]   prev {pid :3d} ({_nm (pid )})  matched current "
            f"{best_cur } ({_nm (best_cur )}) F1={best :.3f}  -> skip")
    return uncovered 


@torch .no_grad ()
#------------ Extract frozen z_task features for all samples in the given loader
def extract_features (fast_model ,loader ,device ):
    """[CHANGED] Extract ONLY the frozen global bottleneck z_task """
    fast_model .eval ()
    all_z ,all_labels =[],[]
    for x ,y ,_ in loader :
        x =x .to (device )
        all_z .append (fast_model .encode_ztask_only (x ).cpu ())
        all_labels .append (y )
    return torch .cat (all_z ,dim =0 ),torch .cat (all_labels ,dim =0 )


#------------ Train the linear CIL classifier on frozen z_task features for all classes seen so far
def train_classifier (fast_model ,npy_path ,upto_task ,global_map ,device ):
    """[CHANGED] Linear probe on the frozen global bottleneck z_task only.
    Trained on ALL classes seen so far (tasks 1..upto_task)."""
    print (f"  [Classifier] Extracting features for tasks 1..{upto_task } ...")

    cil_train_loader =build_cil_train_loader (
    npy_path ,upto_task ,global_map ,BATCH_SIZE ,data_fraction =DATA_FRACTION )
    num_cil_classes =upto_task *len (TASK_SPLITS [1 ])

    z ,labels =extract_features (fast_model ,cil_train_loader ,device )
    clf =LinearCILClassifier (num_classes =num_cil_classes ,device =device )
    feat_loader =DataLoader (TensorDataset (z ,labels ),
    batch_size =BATCH_SIZE ,shuffle =True )
    opt =Adam (clf .parameters (),lr =LR_CIL )
    scheduler =ReduceLROnPlateau (opt ,mode ='min',
    factor =LR_SCHEDULE_FACTOR ,
    patience =LR_SCHEDULE_PATIENCE )
    clf .train ()
    for epoch in range (CIL_CLF_EPOCHS ):
        epoch_loss ,num_batches =0.0 ,0 
        for zf ,y in feat_loader :
            zf =zf .to (device );y =y .to (device )
            loss =F .cross_entropy (clf (zf ),y )
            opt .zero_grad ();loss .backward ();opt .step ()
            epoch_loss +=loss .item ();num_batches +=1 
        avg =epoch_loss /max (num_batches ,1 )
        lr =opt .param_groups [0 ]["lr"]
        pct =100 *(epoch +1 )/CIL_CLF_EPOCHS 
        print (f"    [Classifier] {pct :5.1f}%  ep {epoch +1 }/{CIL_CLF_EPOCHS }  "
        f"loss={avg :.6f}  lr={lr :.2g}")
        scheduler .step (avg )
    return clf 


@torch .no_grad ()
#------------ Evaluate the linear CIL classifier on test samples seen up to the given task
def evaluate_classifier (fast_model ,clf ,npy_path ,upto_task ,global_map ,device ):
    """[CHANGED] Evaluate the linear probe on frozen z_task only."""
    fast_model .eval ();clf .eval ()
    cil_test_loader =build_cil_test_loader (
    npy_path ,upto_task ,global_map ,BATCH_SIZE ,data_fraction =DATA_FRACTION )
    z ,labels =extract_features (fast_model ,cil_test_loader ,device )
    feat_loader =DataLoader (TensorDataset (z ,labels ),
    batch_size =BATCH_SIZE ,shuffle =False )
    correct ,total =0 ,0 
    for zf ,y in feat_loader :
        zf =zf .to (device );y =y .to (device )
        pred =clf (zf ).argmax (dim =1 )
        correct +=(pred ==y ).sum ().item ();total +=y .size (0 )
    return 100.0 *correct /total 


#------------ Run the full Phase-2 CM-AE incremental training loop across all tasks
def train_global (part_models ,global_map ,num_classes ,
ntu_bert_bank ,sentence_bank ,start_task =1 ):
    """


    Fast model initialisation per task
    ────────────────────────────────────
    Task 1  : fresh GlobalSTGCNAE (random weights except frozen backbone).

    Task t>1: load the FULL previous task's saved model (global_ae_task{t-1}.pt).
              This carries forward extra_layers, decoder AND bert_proj_frozen —
              not just the encoder — so the decoder is warm rather than randomly
              reinitialised each task.  global_layers and part_models remain
              frozen as set by GlobalSTGCNAE.__init__; requires_grad flags are
              also set correctly by __init__ so no manual adjustments are needed.

    This also makes global_ae_enc_task{N}.pt redundant: we no longer need a
    separate encoder-only checkpoint because the full model is loaded directly.
    """
    print (f"\n{'='*60 }")
    print ("  PHASE 2 — Global Autoencoder Training ")
    print (f"{'='*60 }")


    if SHUFFLE_VARIANT <0 :
        set_global_seed (GLOBAL_SEED )
    else :
        print (f"  [Seed] seeding disabled (shuffle_variant={SHUFFLE_VARIANT } active)")

    rehearsed_prev =set ()
    for task in range (start_task ,MAX_TASK +1 ):
        print (f"\n{'─'*60 }")
        print (f"  TASK {task }")
        print (f"{'─'*60 }")

        if task ==1 :


            fast_model =GlobalSTGCNAE (
            part_models ={p :part_models [p ]for p in NTU_JOINT_GROUPS },
            device =DEVICE ,
            ).to (DEVICE )
            if PROTOGCN_J_CKPT :
                fast_model .load_protogcn_pretrained (PROTOGCN_J_CKPT )
            print (f"  [FastModel] Task 1 — fresh init (global backbone <- ProtoGCN)")
        else :


            fast_model =load_ae_global_model (task -1 ,part_models ,DEVICE )
            print (f"  [FastModel] Task {task } — loaded full model from task {task -1 } "
            f"(carries decoder + bert_proj_frozen + extra_layers)")


        rehearse_ids =[]
        if task >1 :
            print (f"\n  [Step 4] Uncovered-previous-class detection for task {task } ...")
            rehearse_ids =find_uncovered_previous_classes (
            task ,sentence_bank ,NOVELTY_THRESHOLD ,rehearsed_prev )
            rehearsed_prev .update (rehearse_ids )
            print (f"  [Rehearsal] adding {len (rehearse_ids )} previous classes "
            f"at fraction {NOVEL_SAMPLE_FRACTION :.2f}: {rehearse_ids }")
            print (f"  [Rehearsal] cumulative rehearsed set now: "
            f"{sorted (rehearsed_prev )}")

        task_loader =build_task_plus_novel_loader (
        NPY_PATH ,task ,rehearse_ids ,global_map ,BATCH_SIZE ,
        novel_fraction =NOVEL_SAMPLE_FRACTION ,data_fraction =DATA_FRACTION )

        current_task_set =set (TASK_SPLITS [task ])
        novel_set =set (rehearse_ids )
        if CE_CLASS_MODE ==1 :
            ce_class_set =sorted (current_task_set |novel_set )
            ce_desc ="current + novel"
        elif CE_CLASS_MODE ==2 :
            ce_class_set =sorted (novel_set )
            ce_desc ="novel only"
        else :
            ce_class_set =sorted (current_task_set )
            ce_desc ="current only"

        ce_head =None 
        ce_label_map ={}
        if CE_LOSS_WEIGHT >0.0 and ce_class_set :
            ce_label_map ={raw_id :idx for idx ,raw_id in enumerate (ce_class_set )}
            ce_head =CE_Classifier (num_classes =len (ce_class_set ),device =DEVICE )
            print (f"  [CE] mode={CE_CLASS_MODE } ({ce_desc }) classes={len (ce_class_set )} "
            f"ids={ce_class_set }")
        else :
            print (f"  [CE] mode={CE_CLASS_MODE } ({ce_desc }) skipped because no classes")

        print (f"\n  [Step {'1'if task ==1 else '5'}] "
        f"Training global model on task {task } "
        f"({len (TASK_SPLITS [task ])} current + {len (rehearse_ids )} rehearsed previous classes) ...")

        trainable =[p for p in fast_model .parameters ()if p .requires_grad ]
        if ce_head is not None :
            trainable +=list (ce_head .parameters ())
        opt_fast =Adam (trainable ,lr =LR_GLOBAL )
        sch_fast =ReduceLROnPlateau (opt_fast ,mode ='min',
        factor =LR_SCHEDULE_FACTOR ,
        patience =LR_SCHEDULE_PATIENCE )

        fast_model .train ()
        train_global_ae (
        fast_model ,
        task_loader ,
        opt_fast ,
        sch_fast ,
        ntu_bert_bank ,
        task ,
        ce_head ,
        ce_label_map ,
        CE_LOSS_WEIGHT ,
        )

        save_ae_global_model (fast_model ,task )

    print ("\n  [GlobalAE] Global autoencoder training complete.")


#------------ Run the Phase-3 linear CIL classifier training loop across all tasks
def train_classifiers (part_models ,global_map ,num_classes ):
    print (f"\n{'='*60 }")
    print (f"  PHASE 3 — CIL Classifiers  [LINEAR on frozen z_task]")
    print (f"{'='*60 }")


    if SHUFFLE_VARIANT <0 :
        set_global_seed (GLOBAL_SEED )
    else :
        print (f"  [Seed] seeding disabled (shuffle_variant={SHUFFLE_VARIANT } active)")

    for task in range (1 ,MAX_TASK +1 ):
        print (f"\n{'─'*60 }")
        print (f"  TASK {task } — Train classifier")
        print (f"{'─'*60 }")
        fast_model =load_ae_global_model (task ,part_models ,DEVICE )
        clf =train_classifier (fast_model ,NPY_PATH ,task ,global_map ,DEVICE )
        save_cil_classifier (clf ,task )


#------------ Evaluate trained classifiers across all tasks and report per-task accuracy
def evaluate_classifiers (part_models ,global_map ,num_classes ):
    print (f"\n{'='*60 }")
    print (f"  PHASE 3 — Evaluate CIL Classifiers  [LINEAR on frozen z_task]")
    print (f"{'='*60 }")

    cil_accuracies =[]
    for task in range (1 ,MAX_TASK +1 ):
        print (f"\n{'─'*60 }")
        print (f"  TASK {task } — Evaluate classifier")
        print (f"{'─'*60 }")
        fast_model =load_ae_global_model (task ,part_models ,DEVICE )
        num_cil_classes =task *len (TASK_SPLITS [1 ])
        clf =LinearCILClassifier (num_classes =num_cil_classes ,device =DEVICE )
        clf .load_state_dict (
        torch .load (cil_clf_ckpt_path (task ),map_location =DEVICE ))
        clf .eval ()
        acc =evaluate_classifier (
        fast_model ,clf ,NPY_PATH ,task ,global_map ,DEVICE )
        cil_accuracies .append (acc )
        print (f"  [Linear] After task {task }: {acc :.2f}%")

    return cil_accuracies 


#------------ Entry point: Phase-1 PM-AE pretraining, Phase-2 CM-AE training, Phase-3 classification
def main ():
    assert PART_SOURCE in ("ntu","pku"),f"PART_SOURCE must be 'ntu' or 'pku', got '{PART_SOURCE }'"

    global_map =build_global_class_map ()
    num_classes =len (global_map )

    ntu_bert_bank =BertFeatureBank (
    npy_path =NTU_BERT_FEAT_PATH ,
    device =DEVICE ,
    dataset =BERT_KEY ,
    bert_path =BERT_MODEL_PATH ,
    )

    if PART_SOURCE =="pku":
        part_bert_bank =BertFeatureBank (
        npy_path =PKU_BERT_FEAT_PATH ,
        device =DEVICE ,
        dataset ="pku",
        bert_path =BERT_MODEL_PATH ,
        )
    else :
        part_bert_bank =ntu_bert_bank 

    sentence_bank =SentenceBertBank (
    npy_path =NTU_SENT_FEAT_PATH ,
    device =DEVICE ,
    dataset =BERT_KEY ,
    bert_path =BERT_MODEL_PATH ,
    )

    print (f"Dataset / protocol     : {DATASET .upper ()} / {PROTOCOL }")
    print (f"Tasks                  : {MAX_TASK } x {len (TASK_SPLITS [1 ])} classes")
    print (f"NPY path               : {NPY_PATH }")
    print (f"Device                 : {DEVICE }")
    print (f"Global mode            : {GLOBAL_MODE }")
    print (f"Part source            : {PART_SOURCE .upper ()}")
    if PART_SOURCE =="pku":
        print (f"  PKU npy path         : {PKU_NPY_PATH }")
    print (f"S samples per class    : {S_SAMPLES_PER_CLASS }")
    print (f"λ_align                : {LAMBDA_ALIGN }")
    print (f"CE loss weight         : {CE_LOSS_WEIGHT }")
    print (f"CE class mode          : {CE_CLASS_MODE } "
    "(1=current+novel, 2=novel only, 3=current only)")
    print (f"Novelty threshold      : {NOVELTY_THRESHOLD }")
    print (f"Alpha init             : {ALPHA_INIT }")
    print (f"Global seed            : {GLOBAL_SEED }  "
    f"({'applied to Phase 2+3'if GLOBAL_SEED >=0 and SHUFFLE_VARIANT <0 else 'disabled (shuffle active)'if SHUFFLE_VARIANT >=0 else 'disabled'})")
    print (f"Shuffle variant        : {SHUFFLE_VARIANT }  "
    f"({'shuffled class order — seed disabled'if SHUFFLE_VARIANT >=0 else 'original TASK_SPLITS'})")

    print (f"\n{'='*60 }")
    src_label ="NTU RGB+D 60"if PART_SOURCE =="ntu"else "PKU-MMD, all files"
    print (f"  PHASE 1 — Part Pre-training  [source: {src_label }]")
    print (f"{'='*60 }")

    if TRAIN_PARTS :
        part_models ={}
        for part in NTU_JOINT_GROUPS :
            print (f"\n  Training part: {part }")
            if PART_SOURCE =="ntu":
                loader ,_ =create_part_loader_all_classes (
                NPY_PATH ,part ,BATCH_SIZE ,
                samples_per_class =S_SAMPLES_PER_CLASS ,
                data_fraction =DATA_FRACTION ,
                )
            else :
                loader ,_ =create_part_loader_pku (
                PKU_NPY_PATH ,part ,BATCH_SIZE ,
                samples_per_class =S_SAMPLES_PER_CLASS ,
                )
            model =PartSTGCN (part ,DEVICE ).to (DEVICE )
            if PROTOGCN_J_CKPT and PART_LOAD_PROTOGCN :


                model .load_protogcn_pretrained (PROTOGCN_J_CKPT )
            optimizer =Adam (model .parameters (),lr =LR_PART )
            scheduler =ReduceLROnPlateau (optimizer ,mode ='min',
            factor =LR_SCHEDULE_FACTOR ,
            patience =LR_SCHEDULE_PATIENCE )
            train_part (model ,loader ,optimizer ,scheduler ,part_bert_bank )
            save_part_model (model ,part )
            model .eval ()
            for p in model .parameters ():
                p .requires_grad =False 
            part_models [part ]=model 
            print (f"  {part } → pre-trained & frozen.")
    else :
        part_models ={}
        for part in NTU_JOINT_GROUPS :
            print (f"\n  Loading part: {part }")
            part_models [part ]=load_part_model (part ,DEVICE )

    cil_accuracies =[]
    if TRAIN_GLOBAL :
        train_global (part_models ,global_map ,num_classes ,
        ntu_bert_bank ,sentence_bank ,
        start_task =START_TASK )
    else :
        print ("\n  [INFO] Skipping global training; loading saved checkpoints.")

    if TRAIN_CLASSIFIER :
        train_classifiers (part_models ,global_map ,num_classes )
        cil_accuracies =evaluate_classifiers (
        part_models ,global_map ,num_classes )
    else :
        print ("\n  [INFO] Skipping classifier training/evaluation.")

    print (f"\n{'='*60 }")
    print (f"  Dataset / protocol     : {DATASET .upper ()} / {PROTOCOL }")
    print (f"  Mode                   : {GLOBAL_MODE }")
    print (f"  Part source            : {PART_SOURCE .upper ()}")
    print (f"  Data fraction          : {DATA_FRACTION }")
    print (f"  λ_align                : {LAMBDA_ALIGN }")
    print (f"  Classifier mode        : LINEAR on frozen z_task")
    print (f"  Novelty threshold      : {NOVELTY_THRESHOLD }")
    print (f"  Novel sample fraction  : {NOVEL_SAMPLE_FRACTION }")
    print (f"  Alpha init             : {ALPHA_INIT }")
    print (f"  Global seed            : {GLOBAL_SEED }")
    print (f"  Shuffle variant        : {SHUFFLE_VARIANT }")
    if cil_accuracies :
        avg_acc =sum (cil_accuracies )/len (cil_accuracies )
        print (f"  Per-task acc   : {[f'{a :.2f}'for a in cil_accuracies ]}")
        print (f"  Average acc            : {avg_acc :.2f}%")
    else :
        print ("  Per-task acc   : n/a (TRAIN_CLASSIFIER=False)")
    print (f"{'='*60 }")
    print ("\n=== DONE ===")


    if cil_accuracies :
        import json ,datetime 
        if SHUFFLE_VARIANT >=0 :
            run_tag =f"{DATASET }_{PROTOCOL }_shuf{SHUFFLE_VARIANT }"
        elif GLOBAL_SEED >=0 :
            run_tag =f"{DATASET }_{PROTOCOL }_seed{GLOBAL_SEED }"
        else :
            run_tag =f"{DATASET }_{PROTOCOL }"

        result ={
        "run_tag":run_tag ,
        "dataset":DATASET ,
        "protocol":PROTOCOL ,
        "seed":GLOBAL_SEED ,
        "shuffle_variant":SHUFFLE_VARIANT ,
        "lambda_align":LAMBDA_ALIGN ,
        "novelty_threshold":NOVELTY_THRESHOLD ,
        "novel_sample_fraction":NOVEL_SAMPLE_FRACTION ,
        "per_task_acc":cil_accuracies ,
        "average_acc":round (sum (cil_accuracies )/len (cil_accuracies ),4 ),
        "num_tasks":len (cil_accuracies ),
        "timestamp":datetime .datetime .now ().isoformat (),
        }

        os .makedirs ("results",exist_ok =True )
        json_path =os .path .join ("results",f"{run_tag }.json")
        with open (json_path ,"w")as f :
            json .dump (result ,f ,indent =2 )
        print (f"  [Results] saved → {json_path }")


if __name__ =="__main__":
    main ()