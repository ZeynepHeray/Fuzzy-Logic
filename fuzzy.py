import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf

s_temp = np.arange(7, 17, 1)
s_pre = np.arange(1.75, 4.25, 0.25)
s_carb = np.arange(2.0, 6.5, 0.5)

temp_VC = mf.trimf(s_temp, [7, 7, 9])
temp_C = mf.trimf(s_temp, [7, 9, 11])
temp_N = mf.trimf(s_temp, [10, 12, 14])
temp_H = mf.trimf(s_temp, [12, 14, 16])
temp_VH = mf.trimf(s_temp, [13, 16, 16])

pre_VB = mf.trimf(s_pre, [1.75, 1.75, 2.25])
pre_B = mf.trimf(s_pre, [1.75, 2.25, 2.50])
pre_N = mf.trimf(s_pre, [2.25, 2.75, 3.25])
pre_G = mf.trimf(s_pre, [2.50, 3.25, 3.50])
pre_VG = mf.trimf(s_pre, [2.75, 4.00, 4.00])

carb_VB = mf.trimf(s_carb, [2.0, 2.0, 3.0])
carb_B = mf.trimf(s_carb, [2.0, 3.0, 4.0])
carb_N = mf.trimf(s_carb, [3.0, 4.0, 5.0])
carb_G = mf.trimf(s_carb, [4.0, 5.0, 6.0])
carb_VG = mf.trimf(s_carb, [5.0, 6.0, 6.0])

input_temp = 13
input_pre = 3.00

temp_fit_VC = fuzz.interp_membership(s_temp, temp_VC, input_temp)
temp_fit_C = fuzz.interp_membership(s_temp, temp_C, input_temp)
temp_fit_N = fuzz.interp_membership(s_temp, temp_N, input_temp)
temp_fit_H = fuzz.interp_membership(s_temp, temp_H, input_temp)
temp_fit_VH = fuzz.interp_membership(s_temp, temp_VH, input_temp)

pre_fit_VB = fuzz.interp_membership(s_pre, pre_VB, input_pre)
pre_fit_B = fuzz.interp_membership(s_pre, pre_B, input_pre)
pre_fit_N = fuzz.interp_membership(s_pre, pre_N, input_pre)
pre_fit_G = fuzz.interp_membership(s_pre, pre_G, input_pre)
pre_fit_VG = fuzz.interp_membership(s_pre, pre_VG, input_pre)

rule1 = np.fmin(np.fmin(temp_fit_VC, pre_fit_VB), carb_N)
rule2 = np.fmin(np.fmin(temp_fit_VC, pre_fit_B), carb_N)
rule3 = np.fmin(np.fmin(temp_fit_VC, pre_fit_N), carb_G)
rule4 = np.fmin(np.fmin(temp_fit_VC, pre_fit_G), carb_VG)
rule5 = np.fmin(np.fmin(temp_fit_VC, pre_fit_VG), carb_VG)

rule6 = np.fmin(np.fmin(temp_fit_C, pre_fit_VB), carb_B)
rule7 = np.fmin(np.fmin(temp_fit_C, pre_fit_B), carb_G)
rule8 = np.fmin(np.fmin(temp_fit_C, pre_fit_N), carb_G)
rule9 = np.fmin(np.fmin(temp_fit_C, pre_fit_G), carb_G)
rule10 = np.fmin(np.fmin(temp_fit_C, pre_fit_VG), carb_VG)

rule11 = np.fmin(np.fmin(temp_fit_N, pre_fit_VB), carb_B)
rule12 = np.fmin(np.fmin(temp_fit_N, pre_fit_B), carb_N)
rule13 = np.fmin(np.fmin(temp_fit_N, pre_fit_N), carb_N)
rule14 = np.fmin(np.fmin(temp_fit_N, pre_fit_G), carb_G)
rule15 = np.fmin(np.fmin(temp_fit_N, pre_fit_VG), carb_VG)

rule16 = np.fmin(np.fmin(temp_fit_H, pre_fit_VB), carb_B)
rule17 = np.fmin(np.fmin(temp_fit_H, pre_fit_B), carb_B)
rule18 = np.fmin(np.fmin(temp_fit_H, pre_fit_N), carb_N)
rule19 = np.fmin(np.fmin(temp_fit_H, pre_fit_G), carb_N)
rule20 = np.fmin(np.fmin(temp_fit_H, pre_fit_VG), carb_G)

rule21 = np.fmin(np.fmin(temp_fit_VH, pre_fit_VB), carb_VB)
rule22 = np.fmin(np.fmin(temp_fit_VH, pre_fit_B), carb_B)
rule23 = np.fmin(np.fmin(temp_fit_VH, pre_fit_N), carb_N)
rule24 = np.fmin(np.fmin(temp_fit_VH, pre_fit_G), carb_N)
rule25 = np.fmin(np.fmin(temp_fit_VH, pre_fit_VG), carb_G)

out_VB = rule21
out_B = np.fmax(rule6, np.fmax(np.fmax(rule11, rule16), np.fmax(rule17, rule22)))
out_N = np.fmax(np.fmax(np.fmax(rule1, rule2), np.fmax(rule12, rule13)), np.fmax(np.fmax(rule18, rule19),
                                                                                 np.fmax(rule23, rule24)))
out_G = np.fmax(rule3, np.fmax(np.fmax(rule7, rule8), np.fmax(np.fmax(rule9, rule14), np.fmax(rule20, rule25))))
out_VG = np.fmax(np.fmax(rule4, rule5), np.fmax(rule10, rule15))

out_carb = np.fmax(out_VB, np.fmax(np.fmax(out_B, out_N), np.fmax(out_G, out_VG)))
defuzzied = fuzz.defuzz(s_carb, out_carb, 'mom')
print(defuzzied)

















