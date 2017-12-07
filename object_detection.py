import cv2
import numpy as np

class ObjectDetection:

	def __init__(self, object_map):
		self.colors = {'man': [200, 72, 72], 'skull': [236,236,236]}
		self.map = object_map


	def blob_detect(self, img, id):
		mask = np.zeros(np.shape(img))
		mask[:,:,0] = self.colors[id][0];
		mask[:,:,1] = self.colors[id][1];
		mask[:,:,2] = self.colors[id][2];

		diff = img - mask
		indxs = np.where(diff == 0)
		diff[np.where(diff < 0)] = 0
		diff[np.where(diff > 0)] = 0
		diff[indxs] = 255
		mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
		mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
		# return diff
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.imshow('image',diff)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		return diff
		#flipped co-ords due to numpy blob detect

	def template_detect(self, img, id):
		template = cv2.imread('templates/' + id + '.png')
		w = np.shape(template)[1]
		h = np.shape(template)[0]
		res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
		threshold = 0.8
		loc = np.where( res >= threshold)
		loc[0].setflags(write=True)
		loc[1].setflags(write=True)
		for i in range(np.shape(loc[0])[0]):
		  loc[0][i] += h/2; loc[1][i] += w/2
		return loc, w, h


	def detect_objects(self,img):


	# def get_binary_mask(self, img, id):


	# def get_all_binary_masks(self, img):


def main():
	obj_map = {'man':0, 'skull':1}
	objDet = ObjectDetection(obj_map)
	img_rgb = cv2.imread('19.png')
	img_score_section = img_rgb[15:20, 55:95, :]
	img_game_section = img_rgb[30:,:,:]

	objDet.blob_detect(img_game_section, 'skull')
	


if __name__ == "__main__":
	main()

