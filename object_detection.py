from __future__ import division
import cv2
import numpy as np

class object_detection:

	def __init__(self):
		self.colors = {'man': [200, 72, 72], 'skull': [236,236,236]}
		self.threshold = {'key':0.8, 'door':0.9, 'ladder':0.8}

	def blob_detect(self, img, id):
		mask = np.zeros(img.shape, dtype = "uint8")
		mask[:,:,0] = self.colors[id][0];
		mask[:,:,1] = self.colors[id][1];
		mask[:,:,2] = self.colors[id][2];

		diff = img - mask
		diff[:,:,0] = diff[:,:,1]
		diff[:,:,2] = diff[:,:,1]
		indxs = np.where(diff == 0)
		diff[np.where(diff < 0)] = 0
		diff[np.where(diff > 0)] = 0
		diff[indxs] = 255
		mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
		mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
		return [diff]
		#flipped co-ords due to numpy blob detect

	def template_detect(self, img, id):

		template = cv2.imread('templates/' + id + '.png')
		
		w = np.shape(template)[1]
		h = np.shape(template)[0]
		res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
		threshold = self.threshold[id]
		loc = np.where( res >= threshold)
		loc[0].setflags(write=True)
		loc[1].setflags(write=True)
		masks = []
		for i in range(np.shape(loc[0])[0]):
			mask = np.zeros(img.shape, dtype = "uint8")
			a = loc[0][i] ; b = loc[1][i] 
			c = loc[0][i] + h; d = loc[1][i] + w
			cv2.rectangle(mask, (b,a), (d,c), (255, 255, 255), -1)
			masks.append(mask)
		return masks


	def detect_objects(self,img):
		object_masks = {}
		object_masks['man'] = self.blob_detect(img,'man')
		object_masks['skull'] = self.blob_detect(img,'skull')
		object_masks['key'] = self.template_detect(img,'key')
		object_masks['door'] = self.template_detect(img,'door')
		object_masks['ladder'] = self.template_detect(img,'ladder')	
		return object_masks

	def get_overlap(self,img,goal_mask):
		man_mask = self.blob_detect(img,'man')
		overlap = cv2.bitwise_and(man_mask[0], goal_mask)
		cv2.imshow('image',overlap)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		bits = np.count_nonzero(overlap)
		total_bits = np.count_nonzero(man_mask)
		return float(bits)/float(total_bits)

	def to_grayscale(img):
		return np.mean(img, axis=2).astype(np.uint8)

	def downsample(img):
		return img[::2, ::2]

	def preprocess(img):
		return to_grayscale(downsample(img))

def main():
	objDet = object_detection()
	img_rgb = cv2.imread('templates/19.png')
	img_score_section = img_rgb[15:20, 55:95, :]
	img_game_section = img_rgb[30:,:,:]
	man_mask = objDet.blob_detect(img_game_section,'man')
	print objDet.get_overlap(img_game_section,man_mask[0])

	# objDet.blob_detect(img_game_section, 'skull')
	# object_masks = objDet.detect_objects(img_game_section)
	# for key in object_masks.keys():
	# 	print key
	# 	for mask in object_masks[key]:
	# 		cv2.imshow('image',mask)
	# 		cv2.waitKey(0)
	# 		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

