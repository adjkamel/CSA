from __future__ import annotations 

import argparse 
import os 
import numpy as np 
import torch 
import torch .nn as nn 
from transformers import BertTokenizer ,BertModel 


BERT_DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
BERT_DIM =768 
MAX_TOKENS =512 


NTU_CLASS_NAMES :dict [int ,str ]={
1 :"drink water",
2 :"eat meal",
3 :"brush teeth",
4 :"brush hair",
5 :"drop",
6 :"pick up",
7 :"throw",
8 :"sit down",
9 :"stand up",
10 :"clapping",
11 :"reading",
12 :"writing",
13 :"tear up paper",
14 :"put on jacket",
15 :"take off jacket",
16 :"put on a shoe",
17 :"take off a shoe",
18 :"put on glasses",
19 :"take off glasses",
20 :"put on a hat or cap",
21 :"take off a hat or cap",
22 :"cheer up",
23 :"hand waving",
24 :"kicking something",
25 :"reach into pocket",
26 :"hopping",
27 :"jump up",
28 :"phone call",
29 :"play with phone or tablet",
30 :"type on a keyboard",
31 :"point to something",
32 :"taking a selfie",
33 :"check time from watch",
34 :"rub two hands together",
35 :"nod head or bow",
36 :"shake head",
37 :"wipe face",
38 :"salute",
39 :"put palms together",
40 :"cross hands in front",
41 :"sneeze or cough",
42 :"staggering",
43 :"falling down",
44 :"headache",
45 :"chest pain",
46 :"back pain",
47 :"neck pain",
48 :"nausea or vomiting",
49 :"fan self",
50 :"punch or slap",
51 :"kicking",
52 :"pushing",
53 :"pat on back",
54 :"point finger at someone",
55 :"hugging",
56 :"giving object to someone",
57 :"touch pocket",
58 :"shaking hands",
59 :"walking towards someone",
60 :"walking apart from someone",
}


NTU_SENTENCE_DESCRIPTIONS :dict [int ,str ]={


1 :("The right hand grasps a container and raises it upward toward the mouth; "
"the elbow flexes and the wrist rotates inward; "
"the torso remains upright in a standing or seated posture; "
"the motion is a single forward arc of the arm ending at the face."),

2 :("The right hand grasps utensils and performs repetitive forward arc motions "
"from a surface upward toward the mouth; "
"the elbow flexes and extends alternately; "
"the torso bends slightly forward; "
"the left hand may hold a bowl steady near the torso."),

3 :("Both hands raise a brush upward to the mouth level; "
"the right hand performs repetitive small circular and sideward oscillations "
"across the teeth; "
"the elbow is held in a flexed raised posture; "
"the torso remains upright with the head tilted slightly forward."),

4 :("The right hand raises a brush upward to the head level; "
"the arm performs repetitive downward and upward strokes along the head; "
"the elbow extends and flexes alternately in an arc above shoulder level; "
"the torso remains upright and the head is held steady."),


5 :("The right hand releases grip and lowers downward; "
"an object falls away from the hand; "
"the arm extends downward and then the hand opens outward; "
"the torso may bend slightly forward as the hand lowers toward ground level."),

6 :("The torso bends forward and downward; "
"both hands lower toward the ground and grasp an object; "
"the knees flex slightly as the torso lowers; "
"the hands then raise the object upward as the torso returns to upright."),

7 :("The right arm swings backward and then extends forward rapidly; "
"the hand releases grip at the end of the forward swing; "
"the torso rotates sideward and the weight shifts forward; "
"the motion is a single fast forward arc of the arm."),


8 :("The knees and hips flex simultaneously; "
"the torso lowers downward in a controlled backward arc; "
"both feet remain flat on the ground as the body descends; "
"the arms may extend forward for balance; "
"the motion ends in a seated posture."),

9 :("The knees and hips extend simultaneously; "
"the torso raises upward in a controlled forward arc; "
"both feet push downward against the ground as the body ascends; "
"the arms may lower downward during the rise; "
"the motion ends in an upright standing posture."),


10 :("Both hands are raised to chest level and then clap together; "
"the palms press contact repeatedly in a bilateral inward motion; "
"the elbows flex and extend alternately; "
"the motion is repetitive and rhythmic at mid-torso height."),


11 :("Both hands hold an object at chest to face level; "
"the torso remains upright or slightly forward; "
"the head tilts downward and the eyes direct forward; "
"no significant arm motion; the posture is static for a prolonged duration."),

12 :("The right hand performs small circular and downward strokes on a surface; "
"the wrist leads the motion with fine repetitive rotations; "
"the elbow is partially flexed at a low height near a surface; "
"the torso bends slightly forward and the left hand holds the surface steady."),

13 :("Both hands grasp an object at chest level and then pull outward bilaterally; "
"the arms extend outward simultaneously in opposite directions; "
"the wrists twist outward; "
"the torso remains upright; the motion is a single fast bilateral outward pull."),


14 :("Both arms raise upward and extend backward then lower into sleeves; "
"the torso leans forward then straightens; "
"the hands pull the jacket forward across the torso; "
"the arms flex and the hands grasp and pull at shoulder level."),

15 :("Both hands grasp the jacket at shoulder level and pull outward and downward; "
"the arms lower as the jacket is removed; "
"the torso leans forward slightly; "
"the hands pull each sleeve downward and release."),

16 :("The torso bends forward and downward significantly; "
"one hand lowers toward the foot level and contacts a shoe; "
"the fingers push the shoe onto the foot; "
"the torso then returns upward to an upright posture."),

17 :("The torso bends forward and downward significantly; "
"one hand lowers toward the foot level and grasps a shoe; "
"the fingers pull the shoe off the foot and raise it upward; "
"the torso then returns upward to an upright posture."),

18 :("Both hands raise upward to face level; "
"the fingers grasp a small object and extend it forward toward the face; "
"the arms flex and the hands place the object on the face near the eyes; "
"the torso remains upright."),

19 :("Both hands raise upward to face level; "
"the fingers grasp an object on the face and pull it outward away from the face; "
"the arms extend outward; "
"the torso remains upright."),

20 :("Both hands raise upward above head level; "
"the hands lower a hat downward onto the head; "
"the elbows extend upward then flex downward as the hat is placed; "
"the torso remains upright."),

21 :("Both hands raise upward to head level; "
"the fingers grasp a hat and raise it upward away from the head; "
"the elbows extend upward; "
"the torso remains upright."),


22 :("One or both arms raise upward above shoulder level; "
"the fists clench and the arms may pump upward repetitively; "
"the torso straightens and may lean slightly backward; "
"the head tilts upward."),

23 :("One or both hands raise to shoulder or above-shoulder level; "
"the hand performs large repetitive sideward and upward wave motions; "
"the wrist and elbow drive the wave arc; "
"the torso remains upright."),


24 :("One leg raises forward or sideward with the knee extended; "
"the foot swings forward in a fast arc contacting an object; "
"the supporting leg remains planted; "
"the torso may lean backward for balance; "
"the motion is a single fast swing of the lower leg."),


25 :("One hand lowers to hip level and reaches inward toward the pocket area; "
"the elbow flexes and the hand moves inward toward the torso; "
"the fingers insert downward into a pocket; "
"the torso remains upright."),


26 :("Both feet leave the ground repetitively in upward and downward bouncing motions; "
"the knees flex and extend alternately driving the body upward; "
"the arms may swing upward for momentum; "
"the torso bounces vertically."),

27 :("Both knees flex deeply and both feet push downward against the ground; "
"the body propels upward in a single fast upward motion; "
"the arms swing upward bilaterally for momentum; "
"the torso extends upward; both feet land downward together."),


28 :("One hand raises upward to face level holding a phone; "
"the elbow flexes and the hand holds the device against the ear; "
"the torso remains upright; "
"the mouth opens and the head may tilt sideward toward the device."),

29 :("Both hands hold a device at mid-torso to chest level; "
"the fingers perform repetitive small tap and swipe motions on the device surface; "
"the elbows remain flexed and close to the torso; "
"the head bends forward and downward toward the device."),

30 :("Both hands lower to a surface level in front of the torso; "
"the fingers perform repetitive small downward tap motions on the surface; "
"the elbows flex and rest near the surface; "
"the torso bends slightly forward and the head tilts downward."),


31 :("One arm extends forward with the index finger pointing outward; "
"the elbow extends and the wrist extends forward; "
"the torso may rotate sideward toward the target; "
"the other arm remains at the side."),

32 :("Both hands hold a device at arm length in front of the face; "
"the elbows extend forward and the arms raise to face level; "
"the torso remains upright; "
"the head tilts slightly backward to view the device screen."),

33 :("One hand raises to wrist level near the chest; "
"the head lowers and the eyes direct downward toward the wrist; "
"the other hand may grasp the wrist from below; "
"the torso remains upright."),


34 :("Both hands are held together at chest level; "
"the palms press contact and perform repetitive circular and forward-backward "
"rubbing motions against each other; "
"the elbows flex and the forearms move in opposition; "
"the torso remains upright."),


35 :("The head performs a downward and upward nod repetitively; "
"the neck flexes forward and extends backward alternately; "
"the torso may bend slightly forward from the waist; "
"the arms remain at the sides."),

36 :("The head rotates sideward left and then sideward right repetitively; "
"the neck rotates bilaterally; "
"the torso remains upright; "
"the arms remain at the sides."),

37 :("One or both hands raise to face level and contact the face surface; "
"the hands perform repetitive downward wipe strokes across the face; "
"the elbows flex and the wrists move downward across the face; "
"the torso remains upright."),


38 :("One hand raises upward to forehead level; "
"the elbow extends outward and upward; "
"the hand contacts the forehead with the palm facing outward; "
"the torso straightens and the posture becomes upright and rigid."),


39 :("Both hands raise upward to chest level and press palms together; "
"the elbows flex and the forearms are held together vertically; "
"the fingers extend upward and press inward bilaterally; "
"the torso remains upright."),

40 :("Both arms cross in front of the torso at chest level; "
"the right arm crosses over the left or both cross simultaneously; "
"the elbows flex and the hands lower to the opposite sides of the torso; "
"the torso remains upright."),


41 :("The torso bends slightly forward; "
"the head lowers and the mouth opens; "
"one or both hands may raise to the mouth or nose area; "
"the upper body shakes briefly with the sneeze or cough motion."),

42 :("The torso sways sideward and the steps are irregular and unsteady; "
"the feet step in uncontrolled sideward directions; "
"the arms may extend outward for balance; "
"the knees flex slightly; the body balance is unstable."),

43 :("The knees flex and the torso lowers rapidly downward; "
"the body collapses to the ground in a fast uncontrolled downward motion; "
"the arms may extend outward attempting to break the fall; "
"the final posture is horizontal or near-ground level."),

44 :("One or both hands raise to forehead or temple level and press or hold; "
"the head may tilt sideward or the torso may lean slightly forward; "
"the posture shows reduced motion and the body remains mostly static; "
"the facial expression indicates discomfort."),

45 :("One or both hands lower to chest level and press or hold the chest area; "
"the torso may lean slightly forward; "
"the breathing appears labored with visible upper body motion; "
"the posture shows reduced motion and the body remains mostly static."),

46 :("One or both hands lower to the lower back and press or hold; "
"the torso may lean slightly backward or to one side; "
"the knees may flex slightly; "
"the posture shows reduced motion and the body remains mostly static."),

47 :("One hand raises to neck level and touches or holds the neck area; "
"the head may tilt sideward toward the touching hand; "
"the torso remains upright; "
"the posture shows reduced motion and the body remains mostly static."),

48 :("The torso bends forward and downward; "
"the head lowers as the mouth opens; "
"the hands may raise to the abdomen or mouth area; "
"the body shows rhythmic forward contraction of the torso."),

49 :("One hand holds an object and performs repetitive outward and sideward "
"waving motions in front of the face; "
"the elbow flexes and the wrist drives the fan motion; "
"the torso remains upright."),


50 :("One arm raises upward and swings forward rapidly with the fist clenched or "
"the palm open; "
"the arm extends forward in a fast single arc toward another person; "
"the torso rotates forward and the weight shifts forward; "
"the motion is a single fast forward strike."),

51 :("One leg raises forward with the knee extended; "
"the foot and lower leg swing forward in a fast arc directed at a target; "
"the supporting leg remains planted; "
"the torso leans backward for balance; "
"the motion is a single fast forward lower-body swing."),

52 :("Both hands contact the torso or arms of another person; "
"both arms extend forward together in a single fast forward push motion; "
"the elbows extend and the weight shifts forward; "
"the torso leans forward as the arms push outward."),

53 :("One hand raises to the upper back level of another person; "
"the palm contacts the back area and performs a downward pat motion; "
"the elbow flexes and the arm reaches forward; "
"the torso may lean slightly forward toward the other person."),

54 :("One arm raises and extends forward toward another person; "
"the index finger extends outward and points at the person; "
"the elbow extends forward; "
"the torso may lean slightly forward and rotate sideward toward the person."),

55 :("Both arms raise and extend forward and outward wrapping around another person; "
"the arms pull the other person inward bilaterally; "
"the torso leans forward into contact with the other person; "
"the motion is a bilateral inward pull at torso level."),

56 :("One hand grasps an object and extends forward toward another person; "
"the arm extends forward and transfers the object to the other person's hand; "
"the torso may lean slightly forward; "
"the grip releases as the other person receives the object."),

57 :("One hand lowers to hip level and contacts the pocket area; "
"the fingers touch or press the pocket; "
"the elbow flexes and the arm moves inward toward the torso; "
"the torso remains upright."),

58 :("Both persons raise their right hands and extend them forward; "
"the hands clasp together and perform an upward and downward pumping motion; "
"the elbows flex and extend alternately; "
"both torsos lean slightly forward toward each other."),

59 :("Both feet step forward repetitively closing the distance toward another person; "
"the legs alternate in forward step motions; "
"the torso remains upright and leans slightly forward; "
"the arms swing forward and backward in opposition to the steps."),

60 :("Both feet step backward or sideward repetitively increasing the distance "
"from another person; "
"the legs alternate in backward or sideward step motions; "
"the torso remains upright; "
"the arms swing forward and backward in opposition to the steps."),
}


PKU_CLASS_NAMES :dict [int ,str ]={
1 :"bow",
2 :"brushing hair",
3 :"brushing teeth",
4 :"check time from watch",
5 :"cheer up",
6 :"clapping",
7 :"cross hands in front",
8 :"drink water",
9 :"eat meal or snack",
10 :"falling down",
11 :"hand waving",
12 :"hopping",
13 :"hugging",
14 :"jump up",
15 :"kicking",
16 :"kicking something",
17 :"make a phone call",
18 :"pat on back of someone",
19 :"pick up",
20 :"playing with phone or tablet",
21 :"point finger at someone",
22 :"point to something",
23 :"punching or slapping",
24 :"pushing",
25 :"put on a hat or cap",
26 :"put on jacket",
27 :"rub two hands",
28 :"salute",
29 :"shake fist",
30 :"shake head",
31 :"shaking hands",
32 :"sit down",
33 :"sitting down",
34 :"sneeze or cough",
35 :"stand up",
36 :"standing up from sitting",
37 :"take off a hat or cap",
38 :"take off jacket",
39 :"take off shoe",
40 :"taking a selfie",
41 :"throw",
42 :"touch back",
43 :"touch chest",
44 :"touch head",
45 :"touch neck",
46 :"typing on a keyboard",
47 :"fan oneself",
48 :"walking towards",
49 :"wear a shoe",
50 :"wipe face",
51 :"writing",
}


NTU120_EXTRA_CLASS_NAMES :dict [int ,str ]={
61 :"put on headphone",
62 :"take off headphone",
63 :"shoot at basket",
64 :"bounce ball",
65 :"tennis bat swing",
66 :"juggle table tennis ball",
67 :"hush",
68 :"flick hair",
69 :"thumb up",
70 :"thumb down",
71 :"make OK sign",
72 :"make victory sign",
73 :"staple book",
74 :"counting money",
75 :"cutting nails",
76 :"cutting paper",
77 :"snap fingers",
78 :"open bottle",
79 :"sniff/smell",
80 :"squat down",
81 :"toss a coin",
82 :"fold paper",
83 :"ball up paper",
84 :"play magic cube",
85 :"apply cream on face",
86 :"apply cream on hand",
87 :"put on bag",
88 :"take off bag",
89 :"put object into bag",
90 :"take object out of bag",
91 :"open a box",
92 :"move heavy objects",
93 :"shake fist",
94 :"throw up cap/hat",
95 :"capitulate",
96 :"cross arms",
97 :"arm circles",
98 :"arm swings",
99 :"run on the spot",
100 :"butt kicks",
101 :"cross toe touch",
102 :"side kick",
103 :"yawn",
104 :"stretch oneself",
105 :"blow nose",
106 :"hit with object",
107 :"wield knife",
108 :"knock over",
109 :"grab stuff",
110 :"shoot with gun",
111 :"step on foot",
112 :"high-five",
113 :"cheers and drink",
114 :"carry object",
115 :"take a photo",
116 :"follow",
117 :"whisper",
118 :"exchange things",
119 :"support somebody",
120 :"rock-paper-scissors",
}


NTU120_CLASS_NAMES :dict [int ,str ]={**NTU_CLASS_NAMES ,**NTU120_EXTRA_CLASS_NAMES }


NTU120_EXTRA_SENTENCE_DESCRIPTIONS :dict [int ,str ]={
61 :"Both hands raise upward toward head level holding a headphone; the hands lower the headphone downward onto the ears; the elbows flex upward and then extend slightly outward; the torso remains upright in a standing or seated posture.",
62 :"Both hands raise upward to head level and grasp a headphone; the hands lift the headphone upward away from the ears and lower it downward; the elbows extend upward and then flex downward; the torso remains upright and static.",
63 :"Both hands raise a ball upward above head level; the arms extend upward and forward as the fingers release the ball toward a basket; the elbows extend and the wrists flex forward; the motion is a single fast upward arc and the weight shifts forward.",
64 :"The right hand presses a ball downward toward the ground; the ball returns upward and the hand presses it downward again; the elbow flexes and extends repetitively; the motion is repetitive and rhythmic at hip level.",
65 :"Both hands grasp a bat and raise it backward and upward; the arms swing forward and sideward across the body; the torso rotates sideward and the weight shifts forward; the motion is a single fast horizontal arc of the arms.",
66 :"The right hand holds a bat flat at chest level; the hand taps a ball upward repetitively with small controlled motions; the elbow flexes slightly and the wrist oscillates upward and downward; the torso remains upright and static.",
67 :"The right hand raises upward toward the face; a single finger extends upward and presses contact against the lips of the mouth; the elbow flexes inward toward the chest; the head tilts slightly forward and the torso remains upright.",
68 :"The right hand raises upward toward the head; the fingers reach into the hair and stroke it backward and outward; the elbow flexes upward and then extends sideward; the head tilts sideward as the hand moves away.",
69 :"The right hand raises upward to chest level and closes into a fist; the thumb extends upward while the other fingers remain closed; the elbow flexes and the wrist rotates inward; the torso remains upright and static.",
70 :"The right hand raises upward to chest level and closes into a fist; the thumb extends downward while the other fingers remain closed; the elbow flexes and the wrist rotates outward and downward; the torso remains upright and static.",
71 :"The right hand raises upward to chest level; the thumb and one finger touch contact forming a small circular shape while the other fingers extend upward; the elbow flexes and the wrist rotates outward; the torso remains upright and static.",
72 :"The right hand raises upward to face level; two fingers extend upward and outward while the other fingers remain closed; the elbow flexes upward toward the shoulder; the torso remains upright and static.",
73 :"The left hand holds a book flat on a surface; the right hand raises slightly upward and presses downward on the book with a single fast controlled motion; the elbow extends downward as the palm presses contact; the torso leans slightly forward over the surface.",
74 :"Both hands hold paper at chest level; the fingers of the right hand press and move the paper repetitively in small controlled motions; the elbows flex and the wrists rotate inward; the head tilts downward toward the hands and the torso remains upright.",
75 :"Both hands are held together at chest level; the fingers of the right hand press a small object against the fingers of the left hand repetitively; the elbows flex inward and the wrists rotate; the head tilts downward toward the hands.",
76 :"The left hand holds paper at chest level; the right hand grasps a small object and presses it repetitively across the paper; the elbows flex and the wrists move forward and backward; the head tilts downward and the torso remains upright.",
77 :"The right hand raises upward to chest level; the thumb and one finger press contact and release with a single fast motion; the elbow flexes and the wrist rotates outward; the torso remains upright and static.",
78 :"The left hand grasps a container at chest level; the right hand grasps the top and rotates it repetitively inward and outward; the elbows flex and the wrists twist; the head tilts downward toward the hands and the torso remains upright.",
79 :"The right hand raises an object upward toward the face; the object is held near the nose and the mouth while the head tilts forward; the elbow flexes inward toward the chest; the torso remains upright and the motion is slight and controlled.",
80 :"Both knees flex and the hips lower downward; the torso leans slightly forward while remaining upright; the arms extend forward at chest level and remain steady; the body descends downward and then raises upward in a single controlled motion.",
81 :"The right hand holds a coin at chest level; the hand raises upward and the fingers release the coin upward; the elbow extends upward and the wrist flexes outward; the head tilts upward following the single fast upward arc.",
82 :"Both hands hold paper flat on a surface at hip level; the hands press downward and move inward toward each other; the elbows flex and the wrists rotate inward; the torso leans slightly forward over the surface.",
83 :"Both hands grasp paper at chest level; the fingers press inward and rotate repetitively closing the paper into a small shape; the elbows flex inward and the palms press contact; the head tilts downward and the torso remains upright.",
84 :"Both hands grasp a small cube at chest level; the fingers and the wrists rotate the cube repetitively inward and outward; the elbows flex and remain close to the torso; the head tilts downward toward the hands.",
85 :"The right hand raises upward toward the face; the fingers touch the face and stroke it repetitively in small circular motions; the elbow flexes upward toward the shoulder; the head remains steady and the torso remains upright.",
86 :"Both hands are held together at chest level; the fingers of one hand stroke the back of the other hand repetitively in small circular motions; the elbows flex inward and the palms press contact; the torso remains upright and static.",
87 :"The right hand grasps a bag and raises it upward toward the shoulder; the arm extends backward as the bag lowers onto the shoulder; the elbow flexes upward and then extends downward; the torso leans slightly forward and then remains upright.",
88 :"The right hand raises upward toward the shoulder and grasps a bag; the arm lifts the bag upward and lowers it forward and downward; the elbow flexes and then extends downward; the torso leans slightly forward as the bag descends.",
89 :"The left hand holds a bag open at hip level; the right hand grasps an object and lowers it downward into the bag; the elbow extends downward and the fingers release grip; the torso leans slightly forward toward the bag.",
90 :"The left hand holds a bag at hip level; the right hand reaches downward into the bag and grasps an object; the arm raises the object upward and the elbow flexes toward the chest; the torso leans slightly forward and then remains upright.",
91 :"Both hands reach forward toward a box on a surface; the hands grasp the top and raise it upward and backward; the elbows flex upward as the wrists rotate outward; the torso leans slightly forward over the surface.",
92 :"Both hands reach downward and grasp an object at hip level; the arms lift the object upward as the knees extend and the torso raises upright; the weight shifts backward and the elbows flex toward the chest; the motion is slow and controlled.",
93 :"The right hand closes into a fist and raises upward to chest level; the arm extends forward and backward repetitively in fast controlled motions; the elbow flexes and extends rhythmically; the torso remains upright and leans slightly forward.",
94 :"The right hand raises a hat upward above head level; the fingers release grip and the hat moves upward away from the hand; the elbow extends upward and the wrist flexes outward; the head tilts upward following the single fast upward motion.",
95 :"Both hands raise upward above shoulder level; the palms open outward and the arms extend upward and sideward; the elbows extend and the shoulders raise upward; the torso remains upright and static.",
96 :"Both arms raise upward to chest level and cross inward in front of the torso; each hand grasps the opposite elbow; the elbows flex inward and the palms press contact; the torso remains upright and static.",
97 :"Both arms extend outward and sideward at shoulder level; the arms rotate forward and backward in large circular motions; the elbows remain extended and the shoulders rotate repetitively; the torso remains upright and static.",
98 :"Both arms extend downward beside the torso; the arms swing forward and backward alternately in large controlled arcs; the elbows remain extended and the shoulders rotate; the motion is repetitive and rhythmic.",
99 :"Both knees raise upward alternately toward the chest; the feet press downward against the ground repetitively without moving forward; the arms swing forward and backward in opposition to the legs; the motion is fast and rhythmic and the torso remains upright.",
100 :"Both knees flex backward alternately raising the feet upward toward the hips; the feet press downward against the ground between motions; the arms swing forward and backward beside the torso; the motion is repetitive and rhythmic.",
101 :"The torso bends forward and downward; one hand reaches downward and crosses inward toward the opposite foot; the knees remain extended and the hips flex forward; the body raises upward and the motion alternates between the left and right sides.",
102 :"The weight shifts onto one leg; the other leg raises upward and extends sideward and outward; the knee extends and the foot moves away from the body; the torso leans sideward in opposition and then remains upright.",
103 :"Both hands raise upward toward the face; the mouth opens as the head tilts backward; one hand covers the mouth while the elbows flex upward; the torso extends slightly backward and the motion is slow and controlled.",
104 :"Both arms raise upward above head level and extend outward; the shoulders raise upward and the torso extends backward; the elbows extend and the head tilts backward; the motion is slow and controlled and then the arms lower downward.",
105 :"The right hand raises paper upward toward the face; the hand covers the nose and the mouth while the head tilts forward and downward; the elbow flexes inward toward the chest; the torso leans slightly forward and remains upright.",
106 :"The right hand grasps an object and raises it upward above shoulder level; the arm swings forward and downward toward another person; the elbow extends and the weight shifts forward; the motion is a single fast downward arc of the arm.",
107 :"The right hand grasps a knife at chest level; the arm swings forward and downward repetitively toward another person; the elbow flexes and extends and the wrist rotates inward; the torso rotates sideward and the weight shifts forward.",
108 :"One hand extends forward and pushes an object sideward and downward; the object moves away from the hand toward the ground; the elbow extends forward and the wrist remains flat; the torso leans forward as the weight shifts forward.",
109 :"The right arm extends forward and downward toward an object; the fingers close and grasp the object with a single fast motion; the elbow flexes and pulls the object backward toward the chest; the torso leans forward and then remains upright.",
110 :"Both hands raise forward to chest level holding a gun; the arms extend forward and remain steady toward another person; the elbows extend and the wrists remain flat; the head tilts slightly forward and the torso remains upright.",
111 :"One leg raises upward and the knee flexes forward; the foot presses downward onto the foot of another person; the weight shifts forward onto the pressing leg; the torso remains upright and leans slightly forward.",
112 :"One hand raises upward above shoulder level toward another person; the palms move toward each other and press contact with a single fast motion; the elbow extends upward and the wrist remains flat; the torso extends upward and remains upright.",
113 :"Both hands raise a container forward to chest level toward another person; the containers touch contact and then the hand raises the container upward toward the mouth; the elbows flex and the wrists rotate inward; the torso remains upright.",
114 :"Both hands grasp an object at hip level and hold it steady against the torso; the elbows flex inward and the arms remain static; the feet step forward repetitively moving the body forward; the torso remains upright and leans slightly backward.",
115 :"Both hands raise a phone upward to face level; the arms extend forward and the hands remain steady; the elbows flex outward and one finger presses downward; the head tilts forward and the torso remains upright.",
116 :"Both feet step forward repetitively behind another person; the legs alternate in forward step motions; the arms swing forward and backward in opposition to the steps; the torso remains upright and leans slightly forward.",
117 :"One hand raises upward toward the face and covers the mouth; the torso leans sideward and forward toward the head of another person; the elbow flexes inward and the shoulders raise slightly upward; the head tilts sideward and the motion is slight and controlled.",
118 :"One hand extends forward holding an object toward another person; the other hand reaches forward and grasps the object; the elbows extend forward and then flex backward toward the chest; the torso leans slightly forward and the weight shifts forward.",
119 :"One arm extends forward and outward toward another person; the hand grasps the arm or the back and pulls upward; the elbow flexes and the torso leans sideward as the weight shifts; the motion is slow and controlled.",
120 :"One hand raises upward to chest level and lowers downward repetitively; the fingers extend or remain closed forming a shape at the end of the motion; the elbow flexes and extends rhythmically; the torso remains upright and static.",
}

NTU120_SENTENCE_DESCRIPTIONS :dict [int ,str ]={
**NTU_SENTENCE_DESCRIPTIONS ,
**NTU120_EXTRA_SENTENCE_DESCRIPTIONS ,
}


DATASET_CLASS_NAMES :dict [str ,dict [int ,str ]]={
"ntu":NTU_CLASS_NAMES ,
"ntu120":NTU120_CLASS_NAMES ,
"pku":PKU_CLASS_NAMES ,
}


DATASET_SENTENCE_DESCRIPTIONS :dict [str ,dict [int ,str ]]={
"ntu":NTU_SENTENCE_DESCRIPTIONS ,
"ntu120":NTU120_SENTENCE_DESCRIPTIONS ,
"pku":PKU_CLASS_NAMES ,
}


#------------ Frozen BERT encoder returning pooler_output embeddings for a list of strings
class BertEncoder (nn .Module ):


    #------------ Implementation of __init__
    def __init__ (self ,bert_path :str ="./bert-base-uncased"):
        super ().__init__ ()
        self .tokenizer =BertTokenizer .from_pretrained (bert_path )
        self .bert =BertModel .from_pretrained (bert_path )
        self .bert .eval ()
        for p in self .bert .parameters ():
            p .requires_grad =False 

    @torch .no_grad ()
    #------------ Implementation of forward
    def forward (self ,texts :list [str ])->torch .Tensor :
        """
        Args:
            texts : list of N strings
        Returns:
            Tensor (N, 768) — pooler outputs, float32, on BERT_DEVICE
        """
        enc =self .tokenizer (
        texts ,
        return_tensors ="pt",
        padding =True ,
        truncation =True ,
        max_length =64 ,
        )
        enc ={k :v .to (BERT_DEVICE )for k ,v in enc .items ()}
        out =self .bert (**enc )
        return out .pooler_output .float ()


#------------ Generate and cache BERT class-name embeddings for the given dataset
def generate_bert_features (
dataset :str ,
out_path :str ,
bert_path :str ="./bert-base-uncased",
batch_size :int =64 ,
)->dict [int ,np .ndarray ]:
    """
    Generate BERT pooler features for every class in *dataset* ('ntu' or 'pku').

    Saves a dict  { action_id: np.float32 array of shape (768,) }  to *out_path*.
    Returns that same dict.
    """
    assert dataset in DATASET_CLASS_NAMES ,f"dataset must be 'ntu' or 'pku', got '{dataset }'"

    class_names =DATASET_CLASS_NAMES [dataset ]
    ids =sorted (class_names .keys ())
    texts =[class_names [i ]for i in ids ]

    print (f"[BERT] Generating features for {len (ids )} {dataset .upper ()} classes "
    f"on {BERT_DEVICE } ...")

    model =BertEncoder (bert_path ).to (BERT_DEVICE )
    model .eval ()

    all_feats :list [np .ndarray ]=[]
    for start in range (0 ,len (texts ),batch_size ):
        batch =texts [start :start +batch_size ]
        feats =model (batch ).cpu ().numpy ()
        all_feats .append (feats )
        print (f"  processed {min (start +batch_size ,len (texts ))}/{len (texts )}")

    feat_array =np .concatenate (all_feats ,axis =0 )

    feat_dict :dict [int ,np .ndarray ]={
    action_id :feat_array [i ].astype (np .float32 )
    for i ,action_id in enumerate (ids )
    }

    out_dir =os .path .dirname (os .path .abspath (out_path ))
    os .makedirs (out_dir ,exist_ok =True )
    np .save (out_path ,feat_dict )
    print (f"[BERT] Saved {len (feat_dict )} features → {out_path }")
    return feat_dict 


#------------ Load pre-computed BERT feature dict from a npy file
def load_bert_features (path :str )->dict [int ,np .ndarray ]:
    """
    Load a previously saved BERT feature dict.
    Returns { action_id (int) : np.float32 array of shape (768,) }.
    """
    d =np .load (path ,allow_pickle =True ).item ()
    print (f"[BERT] Loaded {len (d )} features from {path }")
    return d 


#------------ Lookup table mapping class IDs to cached BERT class-name embeddings
class BertFeatureBank :


    #------------ Implementation of __init__
    def __init__ (
    self ,
    npy_path :str ,
    device :str |torch .device ,
    dataset :str ="ntu",
    bert_path :str ="./bert-base-uncased",
    ):
        self .device =device 
        self .dataset =dataset 
        self .class_names =DATASET_CLASS_NAMES [dataset ]


        if not os .path .exists (npy_path ):
            print (f"[BertFeatureBank] {npy_path } not found — generating ...")
            generate_bert_features (
            dataset =dataset ,
            out_path =npy_path ,
            bert_path =bert_path ,
            )

        feat_dict =load_bert_features (npy_path )


        self ._feat_by_id :dict [int ,torch .Tensor ]={
        action_id :torch .tensor (feat ,dtype =torch .float32 ,device =device )
        for action_id ,feat in feat_dict .items ()
        }


    #------------ Implementation of get_by_id
    def get_by_id (self ,action_id :int )->torch .Tensor :
        """Return (768,) tensor for a single raw action ID."""
        return self ._feat_by_id [action_id ]

    #------------ Implementation of get_batch
    def get_batch (self ,action_ids :torch .Tensor )->torch .Tensor :
        """
        action_ids : (B,) long tensor of raw action IDs (1-indexed).
        Returns    : (B, 768) float32 tensor on self.device.
        """
        rows =[self ._feat_by_id [int (a )]for a in action_ids ]
        return torch .stack (rows ,dim =0 )

    #------------ Implementation of get_all_tensor
    def get_all_tensor (self )->tuple [list [int ],torch .Tensor ]:
        """
        Returns (sorted_ids, feat_matrix) where feat_matrix is (N, 768).
        Useful for nearest-neighbour zero-shot lookup over all seen classes.
        """
        ids =sorted (self ._feat_by_id .keys ())
        mat =torch .stack ([self ._feat_by_id [i ]for i in ids ],dim =0 )
        return ids ,mat 

    #------------ Implementation of class_name
    def class_name (self ,action_id :int )->str :
        return self .class_names .get (action_id ,f"action_{action_id }")

    #------------ Implementation of __len__
    def __len__ (self )->int :
        return len (self ._feat_by_id )


#------------ Generate and cache BERT embeddings for long motion-description sentences
def generate_sentence_bert_features (
dataset :str ,
out_path :str ,
bert_path :str ="./bert-base-uncased",
batch_size :int =16 ,
)->dict [int ,np .ndarray ]:

    assert dataset in DATASET_SENTENCE_DESCRIPTIONS ,f"dataset must be 'ntu' or 'pku', got '{dataset }'"

    descriptions =DATASET_SENTENCE_DESCRIPTIONS [dataset ]
    ids =sorted (descriptions .keys ())
    texts =[descriptions [i ]for i in ids ]

    print (f"[SentenceBERT] Generating sentence features for "
    f"{len (ids )} {dataset .upper ()} classes on {BERT_DEVICE } ...")

    model =BertEncoder (bert_path ).to (BERT_DEVICE )
    model .eval ()

    all_feats :list [np .ndarray ]=[]
    for start in range (0 ,len (texts ),batch_size ):
        batch =texts [start :start +batch_size ]
        feats =model (batch ).cpu ().numpy ()
        all_feats .append (feats )
        print (f"  processed {min (start +batch_size ,len (texts ))}/{len (texts )}")

    feat_array =np .concatenate (all_feats ,axis =0 )

    feat_dict :dict [int ,np .ndarray ]={
    action_id :feat_array [i ].astype (np .float32 )
    for i ,action_id in enumerate (ids )
    }

    out_dir =os .path .dirname (os .path .abspath (out_path ))
    os .makedirs (out_dir ,exist_ok =True )
    np .save (out_path ,feat_dict )
    print (f"[SentenceBERT] Saved {len (feat_dict )} sentence features → {out_path }")
    return feat_dict 


import re as _re 

_SPICE_OBJECTS ={
"torso","hand","hands","arm","arms","elbow","elbows","wrist","wrists",
"head","face","palm","palms","finger","fingers","foot","feet","knee","knees",
"hip","hips","shoulder","shoulders","leg","legs","body","weight","chest",
"object","brush","container","utensil","utensils","surface","bowl","phone",
"tablet","keyboard","paper","jacket","shoe","glasses","hat","cap","watch",
"grip","eye","eyes","mouth","neck","back","ground","pocket",


"bag","ball","basket","bat","book","box","coin","cube","gun","hair",
"headphone","knife","lips","thumb",
}
_SPICE_DIRECTIONS ={
"upward","downward","forward","backward","sideward","inward","outward",
"toward","up","down",
}
_SPICE_ATTRIBUTES ={
"right","left","both","upright","flexed","raised","steady","flat","static",
"slight","slightly","repetitive","rhythmic","fast","single","partial",
"partially","controlled","seated","standing","bent","extended","open","closed",
"small","large","circular","alternate","alternately","bilateral","simultaneously",
}
_SPICE_VERBS ={
"raise","lower","extend","flex","rotate","swing","shift","release","bend",
"pull","push","press","grasp","hold","clap","tilt","lean","turn","twist",
"reach","lift","descend","ascend","oscillate","stroke","tap","wave","point",
"cross","rub","nod","shake","wipe","cover","touch","move","perform","remain",
"flexes","extends","raises","lowers","rotates","swings","shifts","bends",
}
_SPICE_LEMMA ={
"hands":"hand","arms":"arm","elbows":"elbow","wrists":"wrist","palms":"palm",
"fingers":"finger","feet":"foot","knees":"knee","hips":"hip","legs":"leg",
"shoulders":"shoulder","eyes":"eye","utensils":"utensil","motions":"motion",
"raises":"raise","raising":"raise","raised":"raise",
"lowers":"lower","lowering":"lower","lowered":"lower",
"extends":"extend","extending":"extend","extended":"extend",
"flexes":"flex","flexing":"flex","flexed":"flex",
"rotates":"rotate","rotating":"rotate","rotated":"rotate",
"swings":"swing","swinging":"swing","swung":"swing",
"shifts":"shift","shifting":"shift","shifted":"shift",
"releases":"release","releasing":"release","released":"release",
"bends":"bend","bending":"bend","bent":"bend",
"pulls":"pull","pushing":"push","pushes":"push","presses":"press","pressing":"press",
"grasps":"grasp","grasping":"grasp","grasped":"grasp","holds":"hold","holding":"hold",
"tilts":"tilt","leans":"lean","turns":"turn","twists":"twist","reaches":"reach",
"lifts":"lift","descends":"descend","ascends":"ascend","strokes":"stroke",
"taps":"tap","waves":"wave","points":"point","crosses":"cross","rubs":"rub",
"nods":"nod","shakes":"shake","wipes":"wipe","covers":"cover","touches":"touch",
"moves":"move","performs":"perform","remains":"remain","oscillations":"oscillate",
"containers":"container",
}
_SPICE_SYN =[
{"raise","lift","elevate","ascend"},{"lower","descend","drop"},
{"bend","flex"},{"extend","straighten","stretch"},
{"rotate","turn","twist"},{"swing","sway"},{"grasp","grip","hold","clutch"},
{"press","push"},{"hand","palm"},{"foot","feet"},
{"forward","front"},{"upward","up"},{"downward","down"},
]

#------------ Return canonical synonym key for a word using the SPICE synonym table
def _spice_syn_key (w ):
    for s in _SPICE_SYN :
        if w in s :
            return min (s )
    return w 

#------------ Return base form of a word via simple suffix rules
def _spice_lemma (w ):
    return _spice_syn_key (_SPICE_LEMMA .get (w ,w ))

#------------ Classify a token as object, relation, direction, attribute or unknown
def _spice_tokclass (w ):
    lw =_SPICE_LEMMA .get (w ,w )
    if lw in _SPICE_OBJECTS or w in _SPICE_OBJECTS :return "OBJ"
    if lw in _SPICE_VERBS or w in _SPICE_VERBS :return "VERB"
    if w in _SPICE_DIRECTIONS :return "DIR"
    if w in _SPICE_ATTRIBUTES :return "ATTR"
    return None 

#------------ Parse a motion description into a set of fact tuples for similarity comparison
def sentence_to_propositions (sentence ):
    """Parse a sentence into a set of SPICE-style semantic-proposition tuples."""
    props =set ()
    for clause in _re .split (r"[;.,]",sentence .lower ()):
        toks =_re .findall (r"[a-z]+",clause )
        tagged =[(t ,_spice_tokclass (t ))for t in toks ]
        objs =[i for i ,(t ,c )in enumerate (tagged )if c =="OBJ"]
        for t ,c in tagged :
            if c =="OBJ":
                props .add ((_spice_lemma (t ),))
        for i ,(t ,c )in enumerate (tagged ):
            if c =="ATTR"and objs :
                j =min (objs ,key =lambda k :abs (k -i ))
                props .add ((_spice_lemma (tagged [j ][0 ]),_spice_lemma (t )))
        for i ,(t ,c )in enumerate (tagged ):
            if c =="VERB":
                left =[k for k in objs if k <i ]
                if left :
                    props .add ((_spice_lemma (tagged [left [-1 ]][0 ]),_spice_lemma (t )))
                for k in range (i +1 ,min (i +4 ,len (tagged ))):
                    if tagged [k ][1 ]=="DIR":
                        props .add ((_spice_lemma (t ),_spice_lemma (tagged [k ][0 ])))
                    if tagged [k ][1 ]=="OBJ":
                        props .add ((_spice_lemma (t ),_spice_lemma (tagged [k ][0 ])))
                        break 
        for i ,(t ,c )in enumerate (tagged ):
            if c =="DIR"and objs :
                j =min (objs ,key =lambda k :abs (k -i ))
                if abs (j -i )<=4 :
                    props .add ((_spice_lemma (tagged [j ][0 ]),_spice_lemma (t )))
    return props 

#------------ Compute symmetric F1 score between two proposition sets for motion similarity
def spice_f1 (props_a ,props_b ):
    """SPICE F1 between two proposition sets (precision/recall over matched tuples)."""
    if not props_a and not props_b :return 1.0 
    if not props_a or not props_b :return 0.0 
    match =len (props_a &props_b )
    if match ==0 :return 0.0 
    P =match /len (props_a );R =match /len (props_b )
    return 2 *P *R /(P +R )


#------------ Lookup combining BERT sentence embeddings with SPICE proposition sets for selective replay
class SentenceBertBank :


    #------------ Implementation of __init__
    def __init__ (
    self ,
    npy_path :str =None ,
    device :str |torch .device ="cpu",
    dataset :str ="ntu",
    bert_path :str ="./bert-base-uncased",
    verbose :bool =True ,
    ):
        self .device =device 
        self .dataset =dataset 


        sentences =DATASET_SENTENCE_DESCRIPTIONS .get (dataset ,{})
        names =(NTU_CLASS_NAMES if dataset =="ntu"
        else globals ().get ("PKU_CLASS_NAMES",{}))
        ids =sorted (set (sentences )|set (names ))
        self ._text ={i :sentences .get (i ,names .get (i ,""))for i in ids }
        self ._props ={i :sentence_to_propositions (self ._text [i ])for i in ids }

        if verbose :
            n_sent =sum (1 for i in ids if i in sentences )
            print (f"[SentenceBertBank/SPICE] {len (ids )} {dataset .upper ()} classes "
            f"({n_sent } with full sentences)  |  novelty = SPICE F1 on raw text")


    #------------ Implementation of get_by_id
    def get_by_id (self ,action_id ):return self ._text [action_id ]
    #------------ Implementation of get_propositions
    def get_propositions (self ,action_id ):return self ._props [action_id ]

    #------------ Implementation of cosine_similarity_matrix
    def cosine_similarity_matrix (self ,ids_a ,ids_b ):
        """(M,N) tensor of SPICE F1 (name kept for drop-in compatibility)."""
        M =torch .zeros (len (ids_a ),len (ids_b ),dtype =torch .float32 ,device =self .device )
        for i ,a in enumerate (ids_a ):
            pa =self ._props [a ]
            for j ,b in enumerate (ids_b ):
                M [i ,j ]=spice_f1 (pa ,self ._props [b ])
        return M 

    #------------ Implementation of _baseline
    def _baseline (self ,previous_ids ):
        if len (previous_ids )<2 :
            return None 
        S =self .cosine_similarity_matrix (previous_ids ,previous_ids ).clone ()
        S .fill_diagonal_ (-2.0 )
        nn =S .max (1 ).values 
        med =nn .median ().item ()
        mad =1.4826 *(nn -nn .median ()).abs ().median ().item ()
        if mad <1e-6 :
            mad =max (nn .std ().item (),1e-6 )
        return med ,mad ,nn 

    #------------ Implementation of novel_classes
    def novel_classes (self ,current_ids ,previous_ids ,threshold =0.45 ):
        """A current class is NOVEL if its best SPICE F1 to any previous class is
        below `threshold`. threshold may be a float or 'auto' (median + 1*mad of
        the known-vs-known F1 baseline)."""
        if not previous_ids :
            return list (current_ids )
        sim =self .cosine_similarity_matrix (current_ids ,previous_ids )
        s =sim .max (1 ).values 
        if threshold =="auto":
            base =self ._baseline (previous_ids )
            thr =(base [0 ]+1.0 *base [1 ])if base else 0.45 
        else :
            thr =threshold 
        return [current_ids [i ]for i in range (len (current_ids ))if s [i ].item ()<thr ]

    #------------ Implementation of suggest_threshold
    def suggest_threshold (self ,current_ids ,previous_ids ):
        """Diagnostic: prints the known-vs-known F1 baseline and every current
        #------------ Implementation of ranked
        class ranked by nearest-previous SPICE F1, so a threshold is easy to pick."""
        print ("\n=== SentenceBertBank(SPICE).suggest_threshold ===")
        base =self ._baseline (previous_ids )
        if base is not None :
            med ,mad ,nn =base 
            print (f"  known-vs-known F1: median={med :.3f} mad={mad :.3f} "
            f"range[{nn .min ():.3f},{nn .max ():.3f}]")
        if not previous_ids :
            print ("  (no previous classes — all current are novel)")
            return list (current_ids )
        sim =self .cosine_similarity_matrix (current_ids ,previous_ids )
        s =sim .max (1 ).values 
        order =torch .argsort (s )
        for i in order .tolist ():
            print (f"    class {current_ids [i ]:3d}  nearest_prev_F1={s [i ].item ():.3f}")
        return None 

    #------------ Implementation of __len__
    def __len__ (self ):
        return len (self ._props )


if __name__ =="__main__":
    parser =argparse .ArgumentParser (
    description ="Pre-generate BERT class-name or sentence features for NTU or PKU."
    )
    parser .add_argument (
    "--dataset",
    choices =["ntu","ntu120","pku"],
    required =True ,
    help ="Dataset whose class names to encode ('ntu' = NTU-60).",
    )
    parser .add_argument (
    "--out_path",
    type =str ,
    required =True ,
    help ="Output .npy path, e.g. data/ntu/bert_feats.npy",
    )
    parser .add_argument (
    "--bert_path",
    type =str ,
    default ="./bert-base-uncased",
    help ="Path to local bert-base-uncased directory.",
    )
    parser .add_argument ("--batch_size",type =int ,default =64 )
    parser .add_argument (
    "--mode",
    choices =["names","sentences"],
    default ="names",
    help ="'names' encodes short class-name strings (default). "
    "'sentences' encodes the long motion-descriptive sentences "
    "(used for inter-task similarity comparison only).",
    )
    args =parser .parse_args ()

    if args .mode =="sentences":
        generate_sentence_bert_features (
        dataset =args .dataset ,
        out_path =args .out_path ,
        bert_path =args .bert_path ,
        batch_size =min (args .batch_size ,16 ),
        )
    else :
        generate_bert_features (
        dataset =args .dataset ,
        out_path =args .out_path ,
        bert_path =args .bert_path ,
        batch_size =args .batch_size ,
        )
