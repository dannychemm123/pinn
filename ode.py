import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#some plt settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 100


# creating tensorflow model

def create_model():
    model = {
        'Dense1':tf.keras.layers.Dense(50, activation='tanh'),
        'Dense2':tf.keras.layers.Dense(50, activation='tanh'),
        'Dense3':tf.keras.layers.Dense(50, activation='tanh'),
        'output_layer':tf.keras.layers.Dense(1)
    }
    
    return model

def call_model(model, x):
    x = model['Dense1'](x)
    x = model['Dense2'](x)
    x = model['Dense3'](x)
    x = model['output_layer'](x)
    
    return x


def pde(x,model):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_pred = call_model(model, x)
        y_x = tape.gradient(y_pred, x)
    y_xx = tape.gradient(y_x, x)
    del tape

    return y_xx + np.pi**2 * tf.sin(np.pi * x) 



# Function to compute the loss
def loss(model,x,x_bc,y_bc):
    res = pde(x, model)
    loss_pde = tf.reduce_mean(tf.square(res))
    y_bc_pred = call_model(model, x_bc)
    loss_bc = tf.reduce_mean(tf.square(y_bc - y_bc_pred))
    return loss_pde + loss_bc

# Function to train the model
def train_step(model,x, x_bc,y_bc,optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(model,x,x_bc,y_bc)
    grads = tape.gradient(loss_value, [layer.trainable_variables for layer in model.values()])
    grads = [item for sublist in grads for item in sublist]  # Flatten the list of lists
    variables = [var for layer in model.values() for var in layer.trainable_variables]
    optimizer.apply_gradients(zip(grads, variables))
    return loss_value

# setting up the problem

x_train = np.linspace(-1, 1, 100).reshape(-1, 1)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

# boundary conditions
x_bc = np.array([[-1.0], [1.0]], dtype=np.float32)
y_bc = np.array([[0.0], [0.0]], dtype=np.float32)
x_bc = tf.convert_to_tensor(x_bc, dtype=tf.float32)
y_bc = tf.convert_to_tensor(y_bc, dtype=tf.float32)
    
# Define the PINN model

model = create_model()

# Define the optimizer with a learning rate scheduler 
lr_sheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sheduler)
# Training the model
epochs = 4000
for epoch in range(epochs):
    loss_value = train_step(model, x_train, x_bc, y_bc, optimizer)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss_value.numpy()}')


#predicting the solution
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_pred = call_model(model, x_test)


# Analytical solution for comparison

y_true = np.sin(np.pi * x_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_pred, label='PINN Prediction', color='blue')
plt.plot(x_test, y_true, label='Analytical Solution', color='red', linestyle='--')
plt.title('PINN vs Analytical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
