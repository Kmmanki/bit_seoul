weight = 0.5
input = 0.5
goal_prediction = 0.8

lr = 0.00001 # 0.1/ 1/ 0.0001/ 10

for i in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction) **2

    print('Error : ' + str(error)+ '\t Prediction : '+ str(prediction))

    up_prediction = input * (weight * lr)
    up_error = (goal_prediction - up_prediction ) ** 2
    
    down_prediction = input *(weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if down_error < up_error :
        weight = weight -lr
    if down_error > up_error :
        weight = weight + lr

'''
#0.0001
Error : 0.36584352249999275      Prediction : 0.19515000000000604
Error : 0.36590400999999273      Prediction : 0.19510000000000605
Error : 0.36596450249999274      Prediction : 0.19505000000000605
Error : 0.3660249999999927       Prediction : 0.19500000000000606

# 0.001
Error : 0.6392002500000004       Prediction : 0.0004999999999997814
Error : 0.6384010000000004       Prediction : 0.0009999999999997814
Error : 0.6392002500000004       Prediction : 0.0004999999999997814
Error : 0.6384010000000004       Prediction : 0.0009999999999997814

#0.01
Error : 0.6320250000000003       Prediction : 0.004999999999999846
Error : 0.6241000000000002       Prediction : 0.009999999999999846
Error : 0.6320250000000003       Prediction : 0.004999999999999846
Error : 0.6241000000000002       Prediction : 0.009999999999999846

#1
Error : 0.0025000000000000044    Prediction : 0.75
Error : 0.20249999999999996      Prediction : 1.25
Error : 0.0025000000000000044    Prediction : 0.75
Error : 0.20249999999999996      Prediction : 1.25

# 10
Error : 19.802500000000002       Prediction : 5.25
Error : 0.30250000000000005      Prediction : 0.25
Error : 19.802500000000002       Prediction : 5.25
Error : 0.30250000000000005      Prediction : 0.25

'''