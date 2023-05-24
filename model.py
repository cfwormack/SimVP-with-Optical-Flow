import torch
from torch import nn
from modules import ConvSC, Inception, Conv2d,Conv3d

# Insert Stride generator

# Insert Encoder



# optical flow addition
class OFPN(nn.Module):
    def __init__(self, C_in, C_out, stride, kern,padding):
        super(OFPN, self).__init__()
        self.ofpn = nn.Sequential(
            Conv2d(C_in, C_out//4, kern, stride, padding, transpose=False, act_norm=False),
            Conv2d(C_out//4, C_out//2, kern, stride, padding, transpose=False, act_norm=False),
            Conv2d(C_out//2, C_in, kern, stride, padding, transpose=False, act_norm=False)

        )
    def forward(self, x):
        ofpn1 = self.ofpn[0](x)
        latent = ofpn1
        for i in range(1, len(self.ofpn)):
            latent = self.ofpn[i](latent)
        return latent

class MEN(nn.Module):
    def __init__(self, C_in, C_out, stride, kern, padding):
        super(MEN, self).__init__()
        self.men = nn.Sequential(
            Conv3d(C_in, C_out//2, kern, stride, padding, transpose=False, act_norm=False),
            # adjust to match the doc and change the stride
            Conv3d(C_out//2, C_out//2, kern, stride, padding, transpose=False, act_norm=False),
            Conv3d(C_out//2, C_in, kern, stride, padding, transpose=False, act_norm=False)

        )

    def forward(self, men):
        for i in range(0,len(self.men)):
            men = self.men[i](men)

        return men

# Insert Decoder

# Insert Mid_Xnet



#here is where I need to connect
class SimVP(nn.Module):                                                                                                                                                         # mnist [10, 3, 64,64 ] KTTI [10,3,128,160]
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8, C_in=64, C_out=128, stride=1, kern=3, padding=1,flow_shape=[10, 3, 64,64]):# taxibj [4,3,32,32] kth [10,3,128,128]
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        Tf,Cf,Hf,Wf=flow_shape
        self.ofpn=OFPN(Cf,C_out,stride,kern,padding) #change cin to the channels try to understand what cin and cout are
        self.men=MEN(Tf,C_out,stride,kern,padding) #-connect to hid shold I convolve around the frames in men rather than the color channels
        self.enc2 = Encoder(Cf, hid_S//2, N_S)
        self.enc = Encoder(C, hid_S//2, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C+Cf, N_S)


    def forward(self, x_raw, x_flows):
        #B-Batch T-frame C-channel H-height W-width
        B, T, C, H, W = x_raw.shape# batch frames channel height width in tensor form
        Bf,Tf,Cf,Hf,Wf= x_flows.shape
        x = x_raw.view(B*T, C, H, W)
        xf=x_flows.view(Bf*Tf,Cf,Hf,Wf)

        embed, skip = self.enc(x)
        inflows=self.ofpn(xf)
        _, C_, H_, W_ = embed.shape
        _f, C_f, H_f, W_f = inflows.shape #encode this information

        z = embed.view(B, T, C_, H_, W_)#C_=64 H_16 W_16 B=16 T=10
        outflows=inflows.view(Bf,Tf, C_f, H_f, W_f )
        flow=self.men(outflows)
        # put encoder here
        _f, Tf2, C_f, H_f, W_f = flow.shape
        flow = flow.view(Bf*Tf2, C_f, H_f, W_f)#16 64 3 64 64
        enc_flow, skip2 = self.enc2(flow)
        _f,  C_f, H_f, W_f = enc_flow.shape

        enc_flow = enc_flow.view(Bf, Tf2, C_f, H_f, W_f)

        frame_and_flow=torch.cat((z,enc_flow), dim=2)
        
        hid = self.hid(frame_and_flow)
        
        hid = hid.reshape(B*T, C_+C_f, H_, W_)
        skips=torch.cat((skip,skip2),dim=1)
        Y = self.dec(hid, skips)
        Y = Y.reshape(B, T, C+Cf, H, W)
        Y,Y_flows=torch.split(Y,split_size_or_sections=[C,3],dim=2)#1 for bw 2 for 2 channel
        return Y
