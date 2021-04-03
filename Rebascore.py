import numpy as np
#inputs
neck=26
leg=58
trunk=41
upper_arm=22
lower_arm=67
wrist=14
Twist_check="Twisted"
bend_check=".notBend"
wrist_check=".Twisted"
shoulder_check =".arm_abducted"
leg_check=".LegRaised"
supported=".leaning"
load=7
shock="ForcelNotActed"
coupling="Acceptable"
Activity="OneRepeat"
#.......
neck_score=0
trunk_score=0
leg_score=0
upper_arm_score=0
lower_arm_score=0
wrist_score=0
# Neck position
if 0 <= neck <= 20 :
    neck_score +=1
else:
    neck_score +=2
# Neck adjust
if Twist_check=="Twisted" :
    neck_score +=1
if bend_check=="Bend" :
    neck_score +=1
print("Neck Score:" ,neck_score)

# Trunk position
if 0 <= trunk <= 1:
    trunk_score +=1
elif trunk <= 20:
    trunk_score +=2
elif 20 < trunk <= 60:
    trunk_score +=3
elif trunk > 60:
    trunk_score +=4
# Trunk adjust
if Twist_check=="Twisted" :
    trunk_score +=1
if bend_check=="Bended" :
    trunk_score +=1
print("Trunk Score:" ,trunk_score) 
# Legs position
leg_score += 1
if leg_check=="LegRaised":
    leg_score += 2
# Legs adjust
if 30 <= leg <= 60:
    leg_score += 1
elif leg > 60:
    leg_score += 2    
print("Leg Score:" ,leg_score)  
# Upper arm position
if 0 <= upper_arm <= 20:
    upper_arm_score +=1
elif 20 < upper_arm <= 45:
    upper_arm_score +=2
elif 45 < upper_arm <= 90:
    upper_arm_score +=3
elif upper_arm > 90:
    upper_arm_score +=4
# Upper arm adjust
if shoulder_check =="shoulder_raised":
    upper_arm_score += 1
elif shoulder_check =="arm_abducted":
    upper_arm_score += 1
if supported == "leaning":
    upper_arm_score -= 1
print("UpperArm Score:" ,upper_arm_score)
# Lower arm position
if 60 <= lower_arm <= 100:
    lower_arm_score += 1
else:
    lower_arm_score += 2
print("LowerArm Score:" ,lower_arm_score)
# Wrist position
if 0 <= wrist <= 15:
    wrist_score += 1
else:
    wrist_score += 2

# Wrist adjust
if wrist_check=="Twisted":
    wrist_score+=1
print("Wrist Score:" ,wrist_score)       

class RebaScore:
    def __init__(self):
        # Table A ( Neck X Trunk X Legs)
        self.table_a = np.zeros((3, 5, 4))
        
        # Init lookup tables
        self.init_table_a()
        self.init_table_b()
        self.init_table_c()


    def init_table_a(self):
        self.table_a = np.array([
                                [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
                                [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
                                [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
                                ])
    def init_table_b(self):
        self.table_b = np.array([
                                [[1, 2, 2], [1, 2, 3]],
                                [[1, 2, 3], [2, 3, 4]],
                                [[3, 4, 5], [4, 5, 5]],
                                [[4, 5, 5], [5, 6, 7]],
                                [[6, 7, 8], [7, 8, 8]],
                                [[7, 8, 8], [8, 9, 9]],
                                ])
    def init_table_c(self):
        self.table_c = np.array([
                                [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
                                [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
                                [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
                                [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
                                [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
                                [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
                                [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
                                [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
                                [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
                                [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
                                [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
                                [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                                ])
    def compute_score_a(self):
        score_a = self.table_a[neck_score-1][trunk_score-1][leg_score-1]
        return score_a
    def compute_score_b(self):
        score_b = self.table_b[upper_arm_score-1][lower_arm_score-1][wrist_score-1]
        return score_b
    def compute_score_c(self, score_a, score_b):
        score_c = self.table_c[score_a-1][score_b-1]
        return score_c
rebaScore = RebaScore()
score_a = rebaScore.compute_score_a()
print("Score A:",score_a)
if 5 <= load <= 10:
    score_a += 1
elif load > 10:
    score_a += 2
elif shock=="ForceActed":
    score_a += 1
score_b= rebaScore.compute_score_b()
print("Score B:",score_b)
if coupling=="Acceptable":
    score_b += 1
elif coupling=="NotAcceptable":
    score_b += 2
elif coupling=="AwkwardUnsafe":
    score_b += 3
score_c = rebaScore.compute_score_c(score_a, score_b)
print("Score C:",score_c)
if Activity=="OneRepeat":
    score_c +=1
elif Activity=="MoreRepeat":
    score_c += 2
elif Activity=="Unstable":
    score_c += 3
print("Reba Score : ", score_c)
