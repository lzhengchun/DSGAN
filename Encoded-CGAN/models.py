import torch, torchvision
import numpy as np

# Channel attention module 
class CAM(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01, r=2):
        super().__init__()
        self.mlp_ops = [
            torch.nn.Linear(in_ch, in_ch//r),
            torch.nn.LeakyReLU(negative_slope=relu_a), 
            torch.nn.Linear(in_ch//r, in_ch),
        ]
        
        self.amlp_layer = torch.nn.Sequential(*self.mlp_ops)
        self.out_act    = torch.nn.Sigmoid()
        
    def forward(self, x, ret_att=False):
        _max_out, _ = torch.max(x, 2, keepdim=False)
        _max_out, _ = torch.max(_max_out, -1, keepdim=False)
        
        _avg_out    = torch.mean(x, 2, keepdim=False)
        _avg_out    = torch.mean(_avg_out, -1, keepdim=False)
        
        _mlp_max    = _max_out
        for layer in self.amlp_layer:
            _mlp_max = layer(_mlp_max)
            
        _mlp_avg    = _avg_out
        for layer in self.amlp_layer:
            _mlp_avg = layer(_mlp_avg)
            
        _attention = self.out_act(_mlp_avg + _mlp_max)
        _attention = _attention.unsqueeze(-1)
        _attention = _attention.unsqueeze(-1)
   
        if ret_att:
            return _attention, _attention * x
        else:
            return _attention * x

# Sparial attention module 
class SAM(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        super().__init__()
        self.cnn_ops = [
            torch.nn.Conv2d(in_channels=2, out_channels=1, \
                            kernel_size=7, padding=3),
            torch.nn.Sigmoid(), ] # use Sigmoid to norm to [0, 1]
        
        self.attention_layer = torch.nn.Sequential(*self.cnn_ops)
        
    def forward(self, x, ret_att=False):
        _max_out, _ = torch.max(x, 1, keepdim=True)
        _avg_out    = torch.mean(x, 1, keepdim=True)
        _out = torch.cat((_max_out, _avg_out), dim=1)
        _attention = _out
        for layer in self.attention_layer:
            _attention = layer(_attention)
           
        if ret_att:
            return _attention, _attention * x
        else:
            return _attention * x

class inception_box(torch.nn.Module):
    def __init__(self, in_ch, o_ch, relu_a=0.01):
        super().__init__()
        assert o_ch % 4 == 0
        self.conv1b1_ops = [
            torch.nn.Conv2d(in_channels=in_ch, out_channels=o_ch//4, kernel_size=1, \
                            stride=1, padding=0),
            torch.nn.LeakyReLU(negative_slope=relu_a), ]
        
        self.conv3b3_ops = [
            torch.nn.Conv2d(in_channels=in_ch, out_channels=o_ch//4, kernel_size=1, \
                            stride=1, padding=0),
            torch.nn.LeakyReLU(negative_slope=relu_a), 
            torch.nn.Conv2d(in_channels=o_ch//4, out_channels=o_ch//4, kernel_size=3, \
                            stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=relu_a), ]
        
        self.conv5b5_ops = [
            torch.nn.Conv2d(in_channels=in_ch, out_channels=o_ch//4, kernel_size=1, \
                            stride=1, padding=0),
            torch.nn.LeakyReLU(negative_slope=relu_a), 
            torch.nn.Conv2d(in_channels=o_ch//4, out_channels=o_ch//4, kernel_size=5, \
                            stride=1, padding=2),
            torch.nn.LeakyReLU(negative_slope=relu_a), ]
        
        self.maxpool_ops = [
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=in_ch, out_channels=o_ch//4, kernel_size=1, \
                            stride=1, padding=0),
            torch.nn.LeakyReLU(negative_slope=relu_a), ]
        
        self.conv1b1 = torch.nn.Sequential(*self.conv1b1_ops)
        self.conv3b3 = torch.nn.Sequential(*self.conv3b3_ops)
        self.conv5b5 = torch.nn.Sequential(*self.conv5b5_ops)
        self.maxpool = torch.nn.Sequential(*self.maxpool_ops)
        
    def forward(self, x): 
        _out_conv1b1 = x
        for layer in self.conv1b1:
            _out_conv1b1 = layer(_out_conv1b1)
            
        _out_conv3b3 = x
        for layer in self.conv3b3:
            _out_conv3b3 = layer(_out_conv3b3)
            
        _out_conv5b5 = x
        for layer in self.conv5b5:
            _out_conv5b5 = layer(_out_conv5b5)
            
        _out_maxpool = x
        for layer in self.conv1b1:
            _out_maxpool = layer(_out_maxpool)
            
        return torch.cat([_out_conv1b1, _out_conv3b3, _out_conv5b5, _out_maxpool], 1)

# sp-cat-ch
class inceContxCSR_SP_Cat_CH(torch.nn.Module):
    def ceil(self, v):
        if v == int(v): return int(v)
        else: return int(v+1)

    def __init__(self, in_ch, ncvar, cvar_ch=8, relu_a=0.01, use_ele=True, cam=False, sam=False, stage_chs=(64, 32, 16)):
        super().__init__()
        self.in_ch = in_ch
        self.norm_chs = 4 * self.ceil(in_ch/4)
        self.use_sam = sam
        self.use_cam = cam
        self.in_norm_ops = [
            torch.nn.Conv2d(in_channels=in_ch, out_channels=self.norm_chs, \
                            kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=self.norm_chs),
            torch.nn.LeakyReLU(negative_slope=relu_a), ]
        
        self.up1_ops = [
            torch.nn.ConvTranspose2d(in_channels=stage_chs[0]+cvar_ch*ncvar, out_channels=stage_chs[0], \
                                         kernel_size=2, stride=2, padding=0),
            torch.nn.LeakyReLU(negative_slope=0.01), ]
        
        self.up2_ops = [
            torch.nn.ConvTranspose2d(in_channels=stage_chs[1], out_channels=stage_chs[1], \
                                         kernel_size=2, stride=2, padding=0),
            torch.nn.LeakyReLU(negative_slope=0.01), ]
        
        if use_ele:
            self.ele_ops = [
                torch.nn.Conv2d(in_channels=1, out_channels=4, \
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01), 
                torch.nn.Conv2d(in_channels=4, out_channels=8, \
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01), ]
        
        self.out_ops = [
            torch.nn.Conv2d(in_channels=stage_chs[2], out_channels=4, \
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.LeakyReLU(negative_slope=0.01), 
            torch.nn.Conv2d(in_channels=4, out_channels=1, \
                            kernel_size=3, stride=1, padding=1),]
        
        self.cvar_inceps = [torch.nn.ModuleList([inception_box(in_ch=1, o_ch=cvar_ch), \
                            inception_box(in_ch=cvar_ch, o_ch=cvar_ch), \
                            inception_box(in_ch=cvar_ch, o_ch=cvar_ch), \
                            inception_box(in_ch=cvar_ch, o_ch=cvar_ch)]) for _ in range(ncvar)]
        self.cvar_inceps = torch.nn.ModuleList(self.cvar_inceps)

        self.ich_layers = torch.nn.Sequential(*self.in_norm_ops)
        
        self.p1_inception1 = inception_box(in_ch = self.norm_chs, o_ch=stage_chs[0])
        self.p1_inception2 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[0])
        self.p1_inception3 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[0])
        self.p1_inception4 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[0])
        self.up1_layers    = torch.nn.Sequential(*self.up1_ops)
        
        self.p2_inception1 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[1])
        self.p2_inception2 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[1])
        self.p2_inception3 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[1])
        self.p2_inception4 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[1])
        self.up2_layers    = torch.nn.Sequential(*self.up2_ops)

        if self.use_cam:
            self.up1_cam = CAM(in_ch = stage_chs[0] + cvar_ch*ncvar)
            self.up2_cam = CAM(in_ch = stage_chs[1])

        if self.use_sam:
            self.up1_sam = SAM(in_ch = stage_chs[0])
            self.up2_sam = SAM(in_ch = stage_chs[1])

        if use_ele:
            self.ele_layers    = torch.nn.Sequential(*self.ele_ops)
            self.p3_inception1 = inception_box(in_ch = 8+stage_chs[1], o_ch = stage_chs[2])
        else:
            self.p3_inception1 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[2])
        self.p3_inception2 = inception_box(in_ch = stage_chs[2], o_ch = stage_chs[2])
        self.p3_inception3 = inception_box(in_ch = stage_chs[2], o_ch = stage_chs[2])
        self.p3_inception4 = inception_box(in_ch = stage_chs[2], o_ch = stage_chs[2])
        self.out_layers    = torch.nn.Sequential(*self.out_ops)
        
    def forward(self, x, cvars, elevation=None, ret_sam=False):
        assert len(cvars) == len(self.cvar_inceps)
        cvar_outs = []
        for _cf, cvar in zip(self.cvar_inceps, cvars):
            _tmp = cvar
            for _f in _cf:
                _tmp = _f(_tmp)
            cvar_outs.append(_tmp)
            
        out_tmp = x
        for layer in self.ich_layers:
            out_tmp = layer(out_tmp) 
            
        out_tmp = self.p1_inception1(out_tmp)
        out_tmp = self.p1_inception2(out_tmp)
        out_tmp = self.p1_inception3(out_tmp)
        out_tmp = self.p1_inception4(out_tmp)        
            
        if self.use_sam: # apply spatial attention 
            if ret_sam:
                atten1, out_tmp = self.up1_sam(out_tmp, ret_att=True) 
            else:
                out_tmp = self.up1_sam(out_tmp) 

        out_tmp = torch.cat([out_tmp,] + cvar_outs, 1) # concat cvars

        if self.use_cam:
            out_tmp = self.up1_cam(out_tmp) # apply channel attention 

        for layer in self.up1_layers:
            out_tmp = layer(out_tmp)  
            
        out_tmp = self.p2_inception1(out_tmp)
        out_tmp = self.p2_inception2(out_tmp)
        out_tmp = self.p2_inception3(out_tmp)
        out_tmp = self.p2_inception4(out_tmp)

        if self.use_cam:
            out_tmp = self.up2_cam(out_tmp) # apply channel attention 
            
        if self.use_sam: # apply spatial attention 
            if ret_sam:
                atten2, out_tmp = self.up2_sam(out_tmp, ret_att=True) 
            else:
                out_tmp = self.up2_sam(out_tmp) 

        for layer in self.up2_layers:
            out_tmp = layer(out_tmp)  
            
        if elevation is not None:
            ele_tmp = elevation
            for layer in self.ele_layers:
                ele_tmp = layer(ele_tmp)  
            out_tmp = torch.cat([out_tmp, ele_tmp], 1)
            
        out_tmp = self.p3_inception1(out_tmp)
        out_tmp = self.p3_inception2(out_tmp)
        out_tmp = self.p3_inception3(out_tmp)
        out_tmp = self.p3_inception4(out_tmp)
        for layer in self.out_layers:
            out_tmp = layer(out_tmp)  
            
        if ret_sam:
            return out_tmp, atten1, atten2
        else:
            return out_tmp
# ch-cat-sp
class inceContxCSR_CH_Cat_SP(torch.nn.Module):
    def ceil(self, v):
        if v == int(v): return int(v)
        else: return int(v+1)

    def __init__(self, in_ch, ncvar, cvar_ch=8, relu_a=0.01, use_ele=True, cam=False, sam=False, stage_chs=(64, 32, 16)):
        super().__init__()
        self.in_ch = in_ch
        self.norm_chs = 4 * self.ceil(in_ch/4)
        self.use_sam = sam
        self.use_cam = cam
        self.in_norm_ops = [
            torch.nn.Conv2d(in_channels=in_ch, out_channels=self.norm_chs, \
                            kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=self.norm_chs),
            torch.nn.LeakyReLU(negative_slope=relu_a), ]
        
        self.up1_ops = [
            torch.nn.ConvTranspose2d(in_channels=stage_chs[0] + cvar_ch*ncvar, out_channels=stage_chs[0], \
                                        kernel_size=2, stride=2, padding=0),
            torch.nn.LeakyReLU(negative_slope=0.01), ]
        
        self.up2_ops = [
            torch.nn.ConvTranspose2d(in_channels=stage_chs[1], out_channels=stage_chs[1], \
                                        kernel_size=2, stride=2, padding=0),
            torch.nn.LeakyReLU(negative_slope=0.01), ]
                
        if use_ele:
            self.ele_ops = [
                torch.nn.Conv2d(in_channels=1, out_channels=4, \
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01), 
                torch.nn.Conv2d(in_channels=4, out_channels=8, \
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01), ]
        
        self.out_ops = [
            torch.nn.Conv2d(in_channels=stage_chs[2], out_channels=4, \
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.LeakyReLU(negative_slope=0.01), 
            torch.nn.Conv2d(in_channels=4, out_channels=1, \
                            kernel_size=3, stride=1, padding=1, bias=False),]
        
        self.cvar_inceps = [torch.nn.ModuleList([inception_box(in_ch=1, o_ch=cvar_ch), \
                            inception_box(in_ch=cvar_ch, o_ch=cvar_ch), \
                            inception_box(in_ch=cvar_ch, o_ch=cvar_ch), \
                            inception_box(in_ch=cvar_ch, o_ch=cvar_ch)]) for _ in range(ncvar)]
        self.cvar_inceps = torch.nn.ModuleList(self.cvar_inceps)

        self.ich_layers = torch.nn.Sequential(*self.in_norm_ops)
        
        self.p1_inception1 = inception_box(in_ch = self.norm_chs, o_ch=stage_chs[0])
        self.p1_inception2 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[0])
        self.p1_inception3 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[0])
        self.p1_inception4 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[0])
        self.up1_layers    = torch.nn.Sequential(*self.up1_ops)
        
        self.p2_inception1 = inception_box(in_ch = stage_chs[0], o_ch = stage_chs[1])
        self.p2_inception2 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[1])
        self.p2_inception3 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[1])
        self.p2_inception4 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[1])
        self.up2_layers    = torch.nn.Sequential(*self.up2_ops)

        if self.use_cam:
            self.up1_cam = CAM(in_ch = stage_chs[0])
            self.up2_cam = CAM(in_ch = stage_chs[1])

        if self.use_sam:
            self.up1_sam = SAM(in_ch = stage_chs[0] + cvar_ch*ncvar)
            self.up2_sam = SAM(in_ch = stage_chs[1])

        if use_ele:
            self.ele_layers    = torch.nn.Sequential(*self.ele_ops)
            self.p3_inception1 = inception_box(in_ch = 8+stage_chs[1], o_ch = stage_chs[2])
        else:
            self.p3_inception1 = inception_box(in_ch = stage_chs[1], o_ch = stage_chs[2])
        self.p3_inception2 = inception_box(in_ch = stage_chs[2], o_ch = stage_chs[2])
        self.p3_inception3 = inception_box(in_ch = stage_chs[2], o_ch = stage_chs[2])
        self.p3_inception4 = inception_box(in_ch = stage_chs[2], o_ch = stage_chs[2])
        self.out_layers    = torch.nn.Sequential(*self.out_ops)
        
    def forward(self, x, cvars, elevation=None, ret_sam=False):
        assert len(cvars) == len(self.cvar_inceps)
        cvar_outs = []
        for _cf, cvar in zip(self.cvar_inceps, cvars):
            _tmp = cvar
            for _f in _cf:
                _tmp = _f(_tmp)
            cvar_outs.append(_tmp)
            
        out_tmp = x
        for layer in self.ich_layers:
            out_tmp = layer(out_tmp) 
            
        out_tmp = self.p1_inception1(out_tmp)
        out_tmp = self.p1_inception2(out_tmp)
        out_tmp = self.p1_inception3(out_tmp)
        out_tmp = self.p1_inception4(out_tmp)        
            
        if self.use_cam:
            out_tmp = self.up1_cam(out_tmp) # apply channel attention 

        out_tmp = torch.cat([out_tmp,] + cvar_outs, 1) # concat cvars

        if self.use_sam: # apply spatial attention 
            if ret_sam:
                atten1, out_tmp = self.up1_sam(out_tmp, ret_att=True) 
            else:
                out_tmp = self.up1_sam(out_tmp) 

        for layer in self.up1_layers:
            out_tmp = layer(out_tmp)  
            
        out_tmp = self.p2_inception1(out_tmp)
        out_tmp = self.p2_inception2(out_tmp)
        out_tmp = self.p2_inception3(out_tmp)
        out_tmp = self.p2_inception4(out_tmp)

        if self.use_cam:
            out_tmp = self.up2_cam(out_tmp) # apply channel attention 
            
        if self.use_sam: # apply spatial attention 
            if ret_sam:
                atten2, out_tmp = self.up2_sam(out_tmp, ret_att=True) 
            else:
                out_tmp = self.up2_sam(out_tmp) 

        for layer in self.up2_layers:
            out_tmp = layer(out_tmp)  
            
        if elevation is not None:
            ele_tmp = elevation
            for layer in self.ele_layers:
                ele_tmp = layer(ele_tmp)  
            out_tmp = torch.cat([out_tmp, ele_tmp], 1)
            
        out_tmp = self.p3_inception1(out_tmp)
        out_tmp = self.p3_inception2(out_tmp)
        out_tmp = self.p3_inception3(out_tmp)
        out_tmp = self.p3_inception4(out_tmp)
        for layer in self.out_layers:
            out_tmp = layer(out_tmp)  
            
        if ret_sam:
            return out_tmp, atten1, atten2
        else:
            return out_tmp
        
class discModel(torch.nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # input layer
        self.operations = [
            torch.nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2), ]
        
        # C128-C256-C512-C512
        out_chs = (128, 256, 512, 512, )
        in_chs  = (64, ) + out_chs[:-1]
        for ic, oc in zip(in_chs, out_chs):
            self.operations += [
                torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=4, \
                                stride=2, padding=1),
                torch.nn.BatchNorm2d(oc),
                torch.nn.LeakyReLU(0.2), ]
            
        # output layers
        self.operations += [
            torch.nn.Conv2d(in_channels=out_chs[-1], out_channels=1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid(), ]
        
        self.layers = torch.nn.Sequential(*self.operations)
        
    def forward(self, x):
        return self.layers(x)