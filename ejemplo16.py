'''
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = (inputs[0] * weights[0]+ inputs[1]* weights[1]+ inputs[2] * weights[2]+bias)
print(output)
'''
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
weights1 = [0.5, -0.91, 0.26, -0.5]
weights2 = [-0.26, -0.27, 0.17, 0.87]

bias = 2
bias1 = 3
bias2 = 0.5

output = [inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias, 
          inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias1,
          inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias2]
print(output)

