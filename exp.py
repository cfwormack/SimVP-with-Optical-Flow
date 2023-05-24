
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *
import cv2 as cv

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)
        #build checkpoint
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        with open('results/Debug/results/Debug/sv/mse_data.txt', 'w') as file:
            file.write('')
        self._build_model()
    #initate simvp model and adds the parameters from the parcer
    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    def _get_data(self):#add data in this function
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)#self.train_flow_loader
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader
    #enable optimizer
    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()#add a mse for the predicted flows /modify mse to

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        #loop over each epoch
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            #load data
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar: # here is where the x 64 by 64 is generated
                self.optimizer.zero_grad()

                batch_flow_x = generate_flows(batch_x).float()
                
                #16 9,3,64,64 vs 16,10,1,64,64 taxibj 8 4 2 32 32
                batch_x, batch_y ,batch_flow_x= batch_x.to(self.device), batch_y.to(self.device),batch_flow_x.to(self.device)
                

                #feed batch of x values to simvp to generate predicted y values
                pred_y = self.model(batch_x,batch_flow_x)#batchx is shape 16,10,1,64,64
                #compute loss by comparing predicted to actual
                loss = self.criterion(pred_y, batch_y)
                #append loss value to list
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            #average the loss
            train_loss = np.average(train_loss)
            #ever so manny epochs perform validation
            if epoch % args.log_step == 0:
                with torch.no_grad():
                    #load validation by calling vali funciton
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                        #validation print statement
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)
        #save model
        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_flow_x = generate_flows(batch_x).float()
            
            batch_x, batch_y, batch_flow_x  = batch_x.to(self.device), batch_y.to(
                self.device), batch_flow_x.to(self.device)
           
            pred_y = self.model(batch_x,batch_flow_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))
            
            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        #save val mse to a text file for graphing.
        with open('results/Debug/results/Debug/sv/mse_data.txt', 'a') as file:
            file.write(str(mse))
            file.write('\n')
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            #noise sensitivity study
            #batch_x=add_noise(batch_x)
            #generate flows
            batch_flow_x = generate_flows(batch_x).float()

            pred_y = self.model(batch_x.to(self.device),batch_flow_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse

# partial credit to https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
def generate_flows(mnist_data):

    #cap = cv.VideoCapture("input_vid.mp4")
    color=False
    video_flows = list([])
    full_list = list([])
    mnist_data=mnist_data.numpy().transpose((0,1,3,4,2))
    for i in range(mnist_data.shape[0]):

        # ret = a boolean return value from
        # getting the frame, first_frame = the
        # first frame in the entire video sequence
        # print(first_frame.shape)

        first_frame = (255 * mnist_data[i, 0]).astype(np.uint8)

        #first_frame = (255*mnist_data[i, 0, 0]).astype(np.uint8)
        # enable grayscale for color images
        if color:
            prev_gray=cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        else:
            prev_gray = first_frame

        # Creates an image filled with zero
        # intensities with the same dimensions
        # as the frame
        #mask = np.zeros_like(maskshape)
        #mask = np.zeros((mnist_data.shape[3], mnist_data.shape[4], 3), dtype="uint8")
        mask = np.zeros((mnist_data.shape[2], mnist_data.shape[3], 3), dtype="uint8")
        #mask=torch.from_numpy(mask)
        # Sets image saturation to maximum
        mask[..., 1] = 255
        # find the region size paramater and fit it to mnist
        # understand what the colors are for
        # try different videos


        for j in range(mnist_data.shape[1]-1):
            # ret = a boolean return value from getting
            # the frame, frame = the current frame being
            # projected in the video
            #ret, frame = cap.read()
            #frame=(255*mnist_data[i, j+1, 0]).to(dtype=torch.uint8)
            #frame = (255 * mnist_data[i, j + 1, 0]).astype(np.uint8)
            frame = (255 * mnist_data[i, j + 1]).astype(np.uint8)
            # Opens a new window and displays the input
            # frame
            if frame is None:
                break
            # Converts each frame to grayscale - we previously
            # only converted the first frame to grayscale
            if color:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Calculates dense optical flow by Farneback method
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                               None,
                                               0.5, 3, 3, 3, 5, 1.2, 0)  # 0.5, 3, 15, 3, 5, 1.2, 0 # for all datasets 0.5, 3, 3, 3, 5, 1.2, 0

            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

            # Sets image hue according to the optical flow
            # direction
            mask[..., 0] = angle * 180 / np.pi / 2

            # Sets image value according to the optical flow
            # magnitude (normalized)
            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

            # Converts HSV to RGB (BGR) color representation
            rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
            # Opens a new window and displays the output frame
            video_flows.append(cv.normalize(rgb, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F))
            #duplicate the last flow and assume that the flow continues this is done to make the dimentions match
            if(j==mnist_data.shape[1]-2):
                video_flows.append(cv.normalize(rgb, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F))

            # Updates previous frame
            prev_gray = gray


        full_list.append(video_flows)

        video_flows = list([])


    full_list = np.array(full_list)
    full_list=torch.from_numpy(full_list)
    full_list=full_list.permute(0,1,4,2,3)
    #print(full_list.shape)
    return full_list

def add_noise(dataset):
    mean = 0
    stddev = 180


    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            # generate gaussian random noise
            noise = np.zeros(dataset[0, 0, 0].shape, np.float32)
            cv.randn(noise, mean, stddev)/255.0
            dataset[i, j,0] = torch.from_numpy(cv.add(dataset[i, j,0].numpy(), noise))

    return dataset
