'''
inputs = [ 1, 2, 3 ]            #  1 ¯¯( 0.2 )¯¯\
weights = [ 0.2, 0.8, -0.5 ]    #  2 --( 0.8 )--- 2.3
bias = 2                        #  3 __(-0.5 )__/  /
                                #      B __( 2 )__/

# Sumatoria de los productos de los inputs y los pesos más el bias
output = (inputs[0] * weights[0] +
          inputs[1] * weights[1] +
          inputs[2] * weights[2] + bias)

print("Ejecicio 01 = ", output) # 2.3
'''
inputs = [ 1, 2, 3, 2.5 ]
weights1 = [ 0.2, 0.8, -0.5, 1 ]
weights2 = [ 0.5, -0.91, 0.26, -0.5]
weights3 = [ -0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
# Neuron 1:
    inputs[0] * weights1[0] +         #    1 ¯ x ¯( 0.2 )¯¯¯\
    inputs[1] * weights1[1] +         #    2 - x -( 0.8 )--- + ------->  4.8
    inputs[2] * weights1[2] +         #    3 _ x _(-0.5 )___/       /
    inputs[3] * weights1[3] + bias1,  #  2.5 _ x _(  1  )__/       +
                                                   #   B __( 2 )__/
# Neuron 2:
    inputs[0] * weights2[0] +         #    1 ¯ x ¯(  0.5  )¯¯¯\
    inputs[1] * weights2[1] +         #    2 - x -( -0.91 )--- + ------->  1.21
    inputs[2] * weights2[2] +         #    3 _ x _(  0.26 )___/       /
    inputs[3] * weights2[3] + bias2,  #  2.5 _ x _( -0.5  )__/       +
                                                     #   B __( 3 )__/
# Neuron 3:
    inputs[0] * weights3[0] +         #    1 ¯ x ¯( -0.26 )¯¯¯\
    inputs[1] * weights3[1] +         #    2 - x -( -0.27 )--- + ------->  2.385
    inputs[2] * weights3[2] +         #    3 _ x _(  0.17 )___/       /
    inputs[3] * weights3[3] + bias3   #  2.5 _ x _(  0.87 )__/       +
                                                   #   B __( 0.5 )__/
]

print("Ejecicio 02 = ", outputs) # [ 4.8, 1.21, 2.385 ]