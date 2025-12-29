import numpy as np
import tensorflow as tf
import keras 
from keras import layers, Sequential

decoder_ring = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L', 'M', 'N', 'O', 'P', 'Q', 
                'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', 
                '8', '9']

model = keras.Sequential()
letter_a = np.array([ 
   [0, 1, 1, 1, 0],
   [1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1],
   [1, 0, 0, 0, 1],
   [1, 0, 0, 0, 1]
])  

letter_b = np.array([
   [1, 1, 1, 0, 0],
   [1, 0, 1, 0, 0],
   [1, 1, 1, 0, 0],
   [1, 0, 1, 0, 0],
   [1, 1, 1, 0, 0]
])  

letter_c = np.array([
   [0, 1, 1, 1, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [0, 1, 1, 1, 0]
])  

letter_d = np.array([
   [0, 0, 1, 1, 1],
   [0, 0, 1, 0, 1],
   [0, 0, 1, 0, 1],
   [0, 0, 1, 0, 1],
   [0, 0, 1, 1, 1]
])  

letter_e = np.array([
   [1, 1, 1, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 1, 1, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 1, 1, 0, 0]
])  

letter_f = np.array([
   [1, 1, 1, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 1, 1, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0]
])  

letter_g = np.array([
   [0, 0, 1, 1, 1],
   [0, 0, 1, 0, 0],
   [0, 0, 1, 1, 1],
   [0, 0, 1, 0, 1],
   [0, 0, 1, 1, 1]
])  

letter_h = np.array([
   [1, 0, 1, 0, 0],
   [1, 0, 1, 0, 0],
   [1, 1, 1, 0, 0],
   [1, 0, 1, 0, 0],
   [1, 0, 1, 0, 0]
])  

letter_i = np.array([
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0]
])  

letter_j = np.array([
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0]
])  

letter_k = np.array([
    [1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0]
])  

letter_l = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0]
])  

letter_m = np.array([ 
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1]
])  

letter_n = np.array([
    [1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1]
])  

letter_o = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0]
])  

letter_p = np.array([
    [0, 0, 1, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])  

letter_q= np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0]
])  

letter_r = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0]
])  

letter_s = np.array([
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0]
])  

letter_t = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])  

letter_u = np.array([
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
])  

letter_v = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])  

letter_w = np.array([
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
])  

letter_x = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])  

letter_y = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])  

letter_z = np.array([
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0]
])  

number_1 = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]
])  

number_2 = np.array([
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0]
])  
number_3 = np.array([
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0]
])  

number_4 = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])  

number_5 = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0]
])  
number_6 = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0]
])  

number_7 = np.array([
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]
])  

number_8 = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0]
])  
number_9 = np.array([
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0]
])  

x = np.array([letter_a, letter_b, 
                letter_c, letter_d, 
                letter_e, letter_f, 
                letter_g, letter_h, 
                letter_i, letter_j, 
                letter_k, letter_l,
                letter_m, letter_n,
                letter_o, letter_p,
                letter_q, letter_r,
                letter_s, letter_t,
                letter_u, letter_v,
                letter_w, letter_x,
                letter_y, letter_z,
                number_1, number_2,
                number_3, number_4,
                number_5, number_6,
                number_7, number_8,
                number_9])
y = np.arange(35)


model.add(layers.Flatten(input_shape=(5, 5, 1)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(38, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)

print("/n--- running final exsam ---")

for i in range(35):
    prediction = model.predict(x[i:i+1])
    predicted_id = np.argmax(prediction)
    predicted_char = decoder_ring[predicted_id]
    print(f"item ID {i}: the AI guessed '{predicted_char}'.")