from object_detection import object_detection
import cv2

class meta_controller:

	def __init__(self):
		self.state = 'start'
		self.key_collected = False
		self.masks = self.load_masks()

	def load_masks(self):
		obj_det = object_detection()
		img_rgb = cv2.imread('templates/19.png')
		img_score_section = img_rgb[15:20, 55:95, :]
		img_game_section = img_rgb[30:,:,:]
		object_masks = obj_det.detect_objects(img_game_section)
		masks = {}
		masks['ladder1'] = object_masks['ladder'][0]
		masks['ladder2'] = object_masks['ladder'][2]
		masks['ladder3'] = object_masks['ladder'][1]
		masks['key'] = 	   object_masks['key'][0]
		masks['door2'] =   object_masks['door'][1]
		# cv2.imshow('img',masks['ladder2'])
		# cv2.waitKey()
		return masks

	def getSubgoal(self):
		if self.key_collected:
			if self.state == 'start':
				return ('door2',self.masks['door2'])
			if self.state == 'ladder1':
				return ('door2',self.masks['door2'])
			if self.state == 'ladder2':
				return ('ladder1',self.masks['ladder1'])
			if self.state == 'ladder3':
				return ('ladder2',self.masks['ladder2'])
			if self.state == 'key':
				return ('ladder3', self.masks['ladder3'])
			return ('door2',self.masks['door2'])
			
		if self.state == 'start':
			return ('ladder1',self.masks['ladder1'])
		if self.state == 'ladder1':
			return ('ladder2',self.masks['ladder2'])
		if self.state == 'ladder2':
			return ('ladder3',self.masks['ladder3'])
		if self.state == 'ladder3':
			return ('key',self.masks['key'])
		if self.state == 'key':
			return ('key', self.masks['key'])
		return ('door2',self.masks['door2'])

	def update_state(self, new_state):
		self.state = new_state

	def getCurrentState(self):
		return self.state

	def got_key(self):
		self.key_collected = True

def main():
	meta = Meta_Controller()
	subgoal, subgoal_image = meta.getSubgoal() 	
	cv2.imshow('image',subgoal_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()


