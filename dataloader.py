
import os 
import re 
import numpy as np 
import torch 
from torch .utils .data import Dataset ,DataLoader 
from natsort import natsorted 
from collections import defaultdict 


#------------ Randomly subsample a fixed fraction of samples from arrays X and y
def _subsample (X ,y ,fraction ,seed =42 ):

    if fraction >=1.0 :
        return X ,y 

    import random 
    from collections import defaultdict as _dd 
    rng =random .Random (seed )
    buckets =_dd (list )
    for i ,label in enumerate (y ):
        buckets [label ].append (i )

    kept =[]
    for label in sorted (buckets ):
        idxs =buckets [label ]
        k =max (1 ,int (len (idxs )*fraction ))
        kept .extend (rng .sample (idxs ,k ))

    kept .sort ()
    X_sub =[X [i ]for i in kept ]
    y_sub =[y [i ]for i in kept ]
    print (f"  [subsample] fraction={fraction :.2f}  "
    f"{len (X_sub )}/{len (X )} samples kept  "
    f"({len (buckets )} classes)")
    return X_sub ,y_sub 


PART_JOINTS ={
"torso":[0 ,1 ,20 ,2 ,3 ],
"left_arm":[4 ,5 ,6 ,7 ,21 ,22 ,20 ],
"right_arm":[8 ,9 ,10 ,11 ,23 ,24 ,20 ],
"left_leg":[12 ,13 ,14 ,15 ,0 ],
"right_leg":[16 ,17 ,18 ,19 ,0 ],
}

NTU_JOINT_GROUPS ={
"torso":[0 ,1 ,20 ,2 ,3 ],
"left_arm":[4 ,5 ,6 ,7 ,21 ,22 ],
"right_arm":[8 ,9 ,10 ,11 ,23 ,24 ],
"left_leg":[12 ,13 ,14 ,15 ],
"right_leg":[16 ,17 ,18 ,19 ],
}


TASK_SPLITS_NTU60 ={
1 :[18 ,40 ,43 ,42 ,49 ,38 ,7 ,50 ,52 ,39 ],
2 :[23 ,41 ,30 ,35 ,33 ,3 ,55 ,37 ,2 ,58 ],
3 :[14 ,11 ,17 ,16 ,13 ,25 ,31 ,1 ,24 ,8 ],
4 :[44 ,53 ,29 ,12 ,9 ,46 ,59 ,36 ,60 ,57 ],
5 :[19 ,54 ,4 ,32 ,56 ,15 ,21 ,51 ,28 ,26 ],
6 :[34 ,48 ,27 ,10 ,5 ,45 ,47 ,20 ,22 ,6 ],
}

TASK_SPLITS_NTU120 ={
1 :[18 ,40 ,43 ,42 ,49 ,38 ,7 ,50 ,52 ,39 ],
2 :[23 ,41 ,30 ,35 ,33 ,3 ,55 ,37 ,2 ,58 ],
3 :[14 ,11 ,17 ,16 ,13 ,25 ,31 ,1 ,24 ,8 ],
4 :[44 ,53 ,29 ,12 ,9 ,46 ,59 ,36 ,60 ,57 ],
5 :[19 ,54 ,4 ,32 ,56 ,15 ,21 ,51 ,28 ,26 ],
6 :[34 ,48 ,27 ,10 ,5 ,45 ,47 ,20 ,22 ,6 ],
7 :[75 ,103 ,91 ,68 ,76 ,106 ,67 ,108 ,72 ,87 ],
8 :[81 ,116 ,96 ,101 ,95 ,93 ,119 ,92 ,117 ,61 ],
9 :[115 ,65 ,110 ,79 ,70 ,100 ,90 ,74 ,109 ,84 ],
10 :[77 ,63 ,105 ,113 ,102 ,118 ,85 ,111 ,64 ,94 ],
11 :[112 ,62 ,82 ,98 ,120 ,71 ,80 ,114 ,104 ,69 ],
12 :[107 ,73 ,97 ,89 ,99 ,83 ,66 ,78 ,86 ,88 ],
}


TASK_SPLITS :dict ={}


NTU60_TRAIN_SUBJECTS =[
1 ,2 ,4 ,5 ,8 ,9 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,25 ,27 ,28 ,31 ,34 ,35 ,38 
]

NTU60_XVIEW_TRAIN_CAMERAS =[2 ,3 ]

NTU120_TRAIN_SUBJECTS =[
1 ,2 ,4 ,5 ,8 ,9 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,25 ,27 ,28 ,31 ,34 ,35 ,38 ,
45 ,46 ,47 ,49 ,50 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,70 ,74 ,78 ,80 ,81 ,
82 ,83 ,84 ,85 ,86 ,89 ,91 ,92 ,93 ,94 ,95 ,97 ,98 ,100 ,103 
]

VALID_PROTOCOLS ={
"ntu60":("xsub","xview"),
"ntu120":("xsub","xset"),
}

DATASET ="ntu60"
PROTOCOL ="xsub"


#------------ Set global train/test split flags for the chosen dataset and evaluation protocol
def configure_protocol (dataset :str ="ntu60",protocol :str ="xsub",verbose :bool =True ):
    """Select dataset + evaluation protocol. Populates TASK_SPLITS in place.
    Returns the number of tasks (6 for NTU-60, 12 for NTU-120)."""
    global DATASET ,PROTOCOL 
    assert dataset in VALID_PROTOCOLS ,f"dataset must be one of {tuple (VALID_PROTOCOLS )}"
    assert protocol in VALID_PROTOCOLS [dataset ],f"protocol '{protocol }' invalid for {dataset }; use {VALID_PROTOCOLS [dataset ]}"
    DATASET ,PROTOCOL =dataset ,protocol 
    splits =TASK_SPLITS_NTU60 if dataset =="ntu60"else TASK_SPLITS_NTU120 
    TASK_SPLITS .clear ()
    TASK_SPLITS .update (splits )
    if verbose :
        n_cls =sum (len (v )for v in TASK_SPLITS .values ())
        print (f"  [Protocol] dataset={dataset }  protocol={protocol }  "
        f"tasks={len (TASK_SPLITS )}  classes={n_cls }")
    return len (TASK_SPLITS )


PKU_NUM_CLASSES =51 
_PKU_ALL_ACTIONS =list (range (1 ,PKU_NUM_CLASSES +1 ))


#------------ Return unit vector of v
def _unit_vector (v ):
    n =np .linalg .norm (v )
    return v if n <1e-8 else v /n 


#------------ Return angle in radians between two 3D vectors
def _angle_between (v1 ,v2 ):
    if np .abs (v1 ).sum ()<1e-6 or np .abs (v2 ).sum ()<1e-6 :
        return 0.0 
    return np .arccos (np .clip (np .dot (_unit_vector (v1 ),_unit_vector (v2 )),-1.0 ,1.0 ))


#------------ Return 3D rotation matrix for angle theta around the given axis
def _rotation_matrix (axis ,theta ):
    if np .abs (axis ).sum ()<1e-6 or np .abs (theta )<1e-6 :
        return np .eye (3 )
    axis =np .asarray (axis ,dtype =np .float64 )
    axis =axis /np .sqrt (np .dot (axis ,axis ))
    a =np .cos (theta /2.0 )
    b ,c ,d =-axis *np .sin (theta /2.0 )
    aa ,bb ,cc ,dd =a *a ,b *b ,c *c ,d *d 
    bc ,ad ,ac ,ab ,bd ,cd =b *c ,a *d ,a *c ,a *b ,b *d ,c *d 
    return np .array ([[aa +bb -cc -dd ,2 *(bc +ad ),2 *(bd -ac )],
    [2 *(bc -ad ),aa +cc -bb -dd ,2 *(cd +ab )],
    [2 *(bd +ac ),2 *(cd -ab ),aa +dd -bb -cc ]])


#------------ ProtoGCN preprocessing: centre on SpineMid and optionally align spine axis
def prenormalize3d (x ,align_center =True ,align_spine =False ,zaxis =(0 ,1 ),xaxis =(8 ,4 )):
    """
    Port of ProtoGCN's PreNormalize3D operating on the loader's array layout.

    Input  x : (C, F, V, M) float32   (C=3, V=25, M=2)
    Output   : (C, F', V, M) float32  (F' = frames after dropping empty ones)
    """
    C ,F ,V ,M =x .shape 
    if not np .any (x ):
        return x 

    sk =np .transpose (x ,(3 ,1 ,2 ,0 )).astype (np .float64 ).copy ()

    index0 =[i for i in range (F )if not np .all (np .isclose (sk [0 ,i ],0 ))]
    if M ==2 :
        index1 =[i for i in range (F )if not np .all (np .isclose (sk [1 ,i ],0 ))]
        if len (index0 )<len (index1 ):
            sk =sk [:,np .array (index1 )]
            sk =sk [[1 ,0 ]]
        else :
            if len (index0 )==0 :
                return x 
            sk =sk [:,np .array (index0 )]
    else :
        if len (index0 )==0 :
            return x 
        sk =sk [:,np .array (index0 )]

    if align_center :
        if sk .shape [2 ]==25 :
            main_center =sk [0 ,0 ,1 ].copy ()
        else :
            main_center =sk [0 ,0 ,-1 ].copy ()
        mask =((sk !=0 ).sum (-1 )>0 )[...,None ]
        sk =(sk -main_center )*mask 

    if align_spine :
        jb ,jt =sk [0 ,0 ,zaxis [0 ]],sk [0 ,0 ,zaxis [1 ]]
        axis =np .cross (jt -jb ,[0 ,0 ,1 ]);ang =_angle_between (jt -jb ,[0 ,0 ,1 ])
        sk =np .einsum ('abcd,kd->abck',sk ,_rotation_matrix (axis ,ang ))
        jr ,jl =sk [0 ,0 ,xaxis [0 ]],sk [0 ,0 ,xaxis [1 ]]
        axis =np .cross (jr -jl ,[1 ,0 ,0 ]);ang =_angle_between (jr -jl ,[1 ,0 ,0 ])
        sk =np .einsum ('abcd,kd->abck',sk ,_rotation_matrix (axis ,ang ))

    return np .transpose (sk ,(3 ,1 ,2 ,0 )).astype (np .float32 )


#------------ Load one NTU or PKU npy skeleton file and return (F,25,3) float32 array
def load_skeleton_sample (path ):

    data =np .load (path ,allow_pickle =True ).item ()

    F =len (data ["skel_body0"])
    x =np .zeros ((3 ,F ,25 ,2 ),dtype =np .float32 )

    x [:,:,:,0 ]=np .stack (data ["skel_body0"]).transpose (2 ,0 ,1 )

    if "skel_body1"in data :
        x [:,:,:,1 ]=np .stack (data ["skel_body1"]).transpose (2 ,0 ,1 )

    x =prenormalize3d (x ,align_center =True ,align_spine =False )
    return x 


#------------ Linearly interpolate skeleton sequence to exactly 64 frames
def resize_sequence_to_64 (x ,target_len =64 ):
    C ,F ,V ,M =x .shape 

    if F ==target_len :
        return x 

    idxs =np .linspace (0 ,F -1 ,target_len )
    new_x =np .zeros ((C ,target_len ,V ,M ),dtype =x .dtype )

    for i ,idx in enumerate (idxs ):
        low ,high =int (np .floor (idx )),int (np .ceil (idx ))
        if low ==high :
            new_x [:,i ]=x [:,low ]
        else :
            w =idx -low 
            new_x [:,i ]=(1 -w )*x [:,low ]+w *x [:,high ]

    return new_x 


#------------ Parse action, subject, setup and camera IDs from an NTU filename
def parse_ntu_filename (fname ):
    setup =int (fname [1 :4 ])
    camera =int (fname [5 :8 ])
    subject =int (fname [9 :12 ])
    action =int (fname [17 :20 ])
    return setup ,camera ,subject ,action 


#------------ Return True if the sample belongs to the training split under the active protocol
def is_train_sample (setup ,camera ,subject ):
    """Protocol-aware train/test membership (uses the module-level DATASET/PROTOCOL
    set by configure_protocol)."""
    if DATASET =="ntu60":
        if PROTOCOL =="xsub":
            return subject in NTU60_TRAIN_SUBJECTS 
        return camera in NTU60_XVIEW_TRAIN_CAMERAS 
    else :
        if PROTOCOL =="xsub":
            return subject in NTU120_TRAIN_SUBJECTS 
        return setup %2 ==0 


#------------ Parse action, person and view IDs from a PKU-MMD Part-II filename
def parse_pku_filename (fname ):
    """
    Parse a PKU .npy/.txt filename → (action, person, view).
    Supports: 'A01N03-L.npy'  and  '1777357782857_A01N03-L.npy'
    """
    base =os .path .splitext (fname )[0 ]
    m =re .search (r'A(\d+)N(\d+)-([LMR])$',base )
    if m :
        return int (m .group (1 )),int (m .group (2 )),m .group (3 )
    raise ValueError (f"Unrecognised PKU filename: '{fname }'")


#------------ PyTorch Dataset wrapping NTU/PKU skeleton npy files with on-the-fly normalisation
class NTUDataset (Dataset ):
    """
    Universal dataset for NTU and PKU skeleton data.

    __getitem__ returns:
        x              : (C, T, V, M)  float32 tensor
        label          : scalar long tensor  (0-indexed CIL label)
        raw_action_id  : scalar long tensor  (original 1-indexed action ID)
    """

    #------------ Implementation of __init__
    def __init__ (self ,X ,y ,raw_ids ,joints =None ,class_map =None ):
        """
        Args:
            X         : list of raw skeleton arrays, each (3, F, 25, 2)
            y         : list of raw action IDs (1-indexed, int)
            raw_ids   : list of raw action IDs (same as y before class_map)
            joints    : optional list of joint indices to select
            class_map : optional dict  raw_id → 0-indexed label
        """
        self .X =X 
        self .y =y 
        self .raw_ids =raw_ids 
        self .joints =joints 
        self .class_map =class_map 

    #------------ Implementation of __len__
    def __len__ (self ):
        return len (self .X )

    #------------ Implementation of __getitem__
    def __getitem__ (self ,idx ):
        x =resize_sequence_to_64 (self .X [idx ])

        if self .joints is not None :
            x =x [:,:,self .joints ,:]

        y =self .y [idx ]
        if self .class_map :
            y =self .class_map [y ]

        raw_id =self .raw_ids [idx ]


        return (
        torch .tensor (x ,dtype =torch .float32 ),
        torch .tensor (y ,dtype =torch .long ),
        torch .tensor (raw_id ,dtype =torch .long ),
        )


#------------ Load training and test tensors for a single CIL task from the NTU npy folder
def build_task_data (npy_path ,task_id ,data_fraction =1.0 ):
    """
    Load NTU train/test splits for a single task.

    Returns:
        train_X, train_y, test_X, test_y
        All y lists contain raw NTU action IDs (1-indexed).
    """
    task_actions =TASK_SPLITS [task_id ]

    train_X ,train_y =[],[]
    test_X ,test_y =[],[]

    for f in natsorted (os .listdir (npy_path )):
        setup ,cam ,subj ,action =parse_ntu_filename (f )

        if action not in task_actions :
            continue 

        x =load_skeleton_sample (os .path .join (npy_path ,f ))

        if is_train_sample (setup ,cam ,subj ):
            train_X .append (x )
            train_y .append (action )
        else :
            test_X .append (x )
            test_y .append (action )

    train_X ,train_y =_subsample (train_X ,train_y ,data_fraction )
    test_X ,test_y =_subsample (test_X ,test_y ,data_fraction )
    return train_X ,train_y ,test_X ,test_y 


#------------ Load all classes from the NTU npy folder for PM-AE pretraining
def build_all_classes_data (npy_path ,samples_per_class =None ,data_fraction =1.0 ):
    """
    Load NTU training data for ALL 60 classes (train subjects only).

    Returns:
        X         : list of raw skeleton arrays
        y         : list of raw NTU action IDs (1-indexed)
        class_map : dict  raw_action_id → contiguous 0-indexed label
    """
    all_actions =[]
    for t in range (1 ,len (TASK_SPLITS )+1 ):
        all_actions .extend (TASK_SPLITS [t ])
    action_set =set (all_actions )

    buckets =defaultdict (list )

    for f in natsorted (os .listdir (npy_path )):
        setup ,cam ,subj ,action =parse_ntu_filename (f )

        if action not in action_set :
            continue 
        if not is_train_sample (setup ,cam ,subj ):
            continue 
        if samples_per_class is not None and len (buckets [action ])>=samples_per_class :
            continue 

        x =load_skeleton_sample (os .path .join (npy_path ,f ))
        buckets [action ].append (x )

    X ,y =[],[]
    for action in all_actions :
        for x in buckets [action ]:
            X .append (x )
            y .append (action )

    X ,y =_subsample (X ,y ,data_fraction )
    class_map ={a :i for i ,a in enumerate (all_actions )}
    return X ,y ,class_map 


#------------ DataLoader over all NTU classes restricted to one body-part subgraph
def create_part_loader_all_classes (npy_path ,part ,batch_size ,
samples_per_class =None ,
data_fraction =1.0 ):
    """
    DataLoader for PartSTGCN pre-training on ALL 60 NTU classes.
    Each batch yields (x, label, raw_action_id).
    """
    X ,y ,class_map =build_all_classes_data (
    npy_path ,samples_per_class ,data_fraction =data_fraction )
    ds =NTUDataset (X ,y ,y ,PART_JOINTS [part ],class_map )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =True ),class_map 


#------------ DataLoader for one task restricted to one body-part subgraph
def create_part_loader (npy_path ,task_id ,part ,batch_size ):
    """Original per-task part loader (kept for compatibility)."""
    Xtr ,Ytr ,Xte ,Yte =build_task_data (npy_path ,task_id )

    class_map ={a :i for i ,a in enumerate (TASK_SPLITS [task_id ])}

    train_ds =NTUDataset (Xtr ,Ytr ,Ytr ,PART_JOINTS [part ],class_map )
    test_ds =NTUDataset (Xte ,Yte ,Yte ,PART_JOINTS [part ],class_map )

    return (
    DataLoader (train_ds ,batch_size =batch_size ,shuffle =True ),
    DataLoader (test_ds ,batch_size =batch_size ,shuffle =False ),
    )


#------------ DataLoader for one task using all 25 joints
def create_full_loader (npy_path ,task_id ,batch_size ):
    Xtr ,Ytr ,Xte ,Yte =build_task_data (npy_path ,task_id )
    train_ds =NTUDataset (Xtr ,Ytr ,Ytr )
    return DataLoader (train_ds ,batch_size =batch_size ,shuffle =True )


#------------ Load PKU-MMD Part-II npy files into tensors for PM-AE pretraining
def build_pku_all_classes_data (npy_path ,samples_per_class =None ):
    """
    Load ALL PKU-MMD .npy files for PartSTGCN pre-training.
    No train/test split.  Every file — all persons, all three views — is used.

    Returns:
        X         : list of raw skeleton arrays, each (3, F, 25, 2)
        y         : list of raw PKU action IDs (1-indexed, 1..51)
        class_map : dict  raw_action_id → contiguous 0-indexed label (0..50)
    """
    action_set =set (_PKU_ALL_ACTIONS )
    buckets =defaultdict (list )

    for fname in natsorted (os .listdir (npy_path )):
        if not fname .endswith (".npy"):
            continue 

        try :
            action ,person ,view =parse_pku_filename (fname )
        except ValueError as e :
            print (f"  [PKU WARN] Skipping '{fname }': {e }")
            continue 

        if action not in action_set :
            continue 

        if samples_per_class is not None and len (buckets [action ])>=samples_per_class :
            continue 

        x =load_skeleton_sample (os .path .join (npy_path ,fname ))
        buckets [action ].append (x )

    X ,y =[],[]
    for action in _PKU_ALL_ACTIONS :
        for x in buckets [action ]:
            X .append (x )
            y .append (action )

    class_map ={a :i for i ,a in enumerate (_PKU_ALL_ACTIONS )}

    found =sum (1 for a in _PKU_ALL_ACTIONS if buckets [a ])
    print (f"  [PKU] {len (X )} samples  |  {found }/{PKU_NUM_CLASSES } classes  "
    f"|  all views, all persons, no train/test split")
    return X ,y ,class_map 


#------------ DataLoader over PKU-MMD Part-II restricted to one body-part subgraph
def create_part_loader_pku (npy_path ,part ,batch_size ,samples_per_class =None ):
    """
    DataLoader for PartSTGCN pre-training on ALL PKU-MMD data.
    Each batch yields (x, label, raw_action_id).

    Returns:
        loader    : DataLoader (shuffled)
        class_map : dict  raw_action_id (1..51) → 0-indexed label (0..50)
    """
    X ,y ,class_map =build_pku_all_classes_data (npy_path ,samples_per_class )
    ds =NTUDataset (X ,y ,y ,PART_JOINTS [part ],class_map )
    loader =DataLoader (ds ,batch_size =batch_size ,shuffle =True )
    print (f"  [PKU] Part '{part }': {len (ds )} samples  |  "
    f"batch_size={batch_size }  |  samples_per_class={samples_per_class }")
    return loader ,class_map 


configure_protocol ("ntu60","xsub",verbose =False )
