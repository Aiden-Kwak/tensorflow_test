import tensorflow as tf

# h = [170, 180, 17, 160]
# shoe = [260, 270, 265, 255]
h = 170
shoe = 260

a = tf.Variable(0.1)
b = tf.Variable(0.2)

 # 경사하강법
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 손실함수
def loss_func():
    pre = h * a + b
    return tf.square(shoe - pre)

for i in range(300):
    opt.minimize(loss_func, var_list=[a,b])
    print(a.numpy(), b.numpy())