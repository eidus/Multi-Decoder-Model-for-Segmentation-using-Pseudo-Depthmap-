import os
import cv2 as cv

class CityScapes():
    def __init__(self, root, mode='train', return_paths=False):
        self.root = root
        self.return_paths = return_paths
        self.mode = mode
        # 파일명 가져오기
        self.file_list = os.path.join(self.root, self.mode +'.txt')
        self.files = [line.rstrip() for line in tuple(open(self.file_list, "r"))]


    def __getitem__(self, idx):
        index = self.files[idx] # index 가져오기

        self.rgb_paths = os.path.join(self.root, 'leftImg8bit', self.mode, index.split('_')[0],f'{index}_leftImg8bit.png')
        self.mask_paths = os.path.join(self.root, 'gtFine', self.mode, index.split('_')[0],f'{index}_gtFine_labelIds.png')
        self.disparity_paths = os.path.join(self.root, 'disparity', self.mode, index.split('_')[0],f'{index}_disparity.png')
        
        rgb_image = cv.cvtColor(cv.imread(self.rgb_paths), cv.COLOR_BGR2RGB)
        mask = cv.imread(self.mask_paths, cv.IMREAD_UNCHANGED)
        disparity = cv.imread(self.disparity_paths, cv.IMREAD_UNCHANGED)

        if self.return_paths:
            return rgb_image, (mask, disparity), self.rgb_paths, (self.mask_paths, self.disparity_paths)

        return rgb_image, (mask, disparity)
    
    def __len__(self):
        return len(self.rgb_paths)