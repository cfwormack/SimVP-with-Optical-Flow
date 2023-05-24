import torch
import numpy as np
from torch.utils.data import Dataset
import hickle as hkl
import cv2 as cv
class KTTIDataset(Dataset):
    def __init__(self, X, Y):
        super(KTTIDataset, self).__init__()
        self.X = cv.normalize(X, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
        self.Y = cv.normalize(Y, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F) #check normalize
        #Y.astype(float) / 255.0
        # X.astype(float) / 255.0
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()

        return data, labels
def split_frames(set):

    X_set=list([])
    Y_set=list([])
    X_frames = list([])
    Y_frames = list([])
    count=0

    for i in range(len(set)):
        if count<=9:
            X_frames.append(set[i])
            count=count+1
        elif count>9:
            Y_frames.append(set[i])
            count = count + 1

        if count ==20:
            count=0
            X_set.append(X_frames)
            Y_set.append(Y_frames)
            X_frames = list([])
            Y_frames = list([])


    return np.array(X_set).transpose((0,1,4,2,3)),np.array( Y_set).transpose((0,1,4,2,3))
def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train = hkl.load(data_root+"kitti/kitti_hkl/X_train.hkl")
    X_train, Y_train =split_frames(train)

    test = hkl.load(data_root + "kitti/kitti_hkl/X_test.hkl")
    X_test, Y_test = split_frames(test)


#2069, 10,128,160,3
    train_set = KTTIDataset(X=X_train, Y=Y_train)
    test_set = KTTIDataset(X=X_test, Y=Y_test)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, None, dataloader_test, 0, 1


#if __name__ == '__main__':
# train_loader, vali_loader, test_loader, data_mean, data_std = load_data(2, 2, '/home/craigw98/Downloads/SimVP-Simpler-yet-Better-Video-Prediction-master/data/',0)
# for x,y in test_loader:
#  print(x[0][0].shape)
#   cv.imshow("Frame 0", np.array(x[0][0]).astype(np.uint8))
#   cv.imshow("Frame 1", np.array(x[0][1]).astype(np.uint8))
#   cv.waitKey(0)
   #print(y.shape)
   #display x