import torch, cv2
from torch.autograd import Variable
import numpy as np
from object_detection import object_detection
import gym
from model.neural_network import neural_network


env = gym.make('MontezumaRevenge-v0')
observation = env.reset()
frame = env.render(mode='rgb_array')
frame = frame[30:,:,:]
frame = frame[::2, ::2]
print frame.shape
objDet = object_detection()
man_mask = objDet.blob_detect(frame,'man')
actor = neural_network(18)
print actor

frame = np.expand_dims(np.mean(frame, axis=2).astype(np.uint8),axis=2)
#print "frame", frame.shape
man_mask = np.squeeze(np.array(man_mask))
man_mask = np.expand_dims(np.mean(man_mask, axis=2).astype(np.uint8), axis=2)
#print "man_mask", man_mask.shape
input_vector = np.concatenate([frame, frame, frame, frame, man_mask], axis=2)
#print input_vector.shape
input_vector = input_vector.reshape((5, -1, 90, 80))
#print input_vector.shape
var = Variable(torch.from_numpy(input_vector).type(torch.FloatTensor))
print max(actor(var).data)
