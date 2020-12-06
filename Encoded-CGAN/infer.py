import numpy as np 
import sys, os, time, argparse, torch, h5py, glob, skimage.transform, shutil
from models import inceContxCSR_SP_Cat_CH as mdlCSR_SP
from models import inceContxCSR_CH_Cat_SP as mdlCSR_CH
from util import *
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-exp',    type=str, required=True, help='experiment name')
parser.add_argument('-logtrs', type=str2bool, default=False, help='log transform')
parser.add_argument('-sam',    type=str2bool, default=True, help='apply spatial attention')
parser.add_argument('-cam',    type=str2bool, default=True, help='apply channel attention')
parser.add_argument('-mdlsz',  type=int, default=256, help='channels of the 1st box')
parser.add_argument('-ckpt',   type=int, default=None, help='epoch of model chpt')
parser.add_argument('-cvars',  type=str2list, default=(), help='vars as condition')
parser.add_argument('-ch1st',  type=str2bool, default=False, help='ch attention first')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

torch_devs = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_mask = np.arange(0, 2911, dtype=np.uint) >= 2328

rain_12km = h5py.File('../dataset/WRF_precip_2005_12km-mask-clip0p05-99p5.hdf5', "r")['rain'][:][test_mask]

vars_all = {}
with h5py.File('../dataset/WRF_50km_vars-mask-clip0p05-99p5.hdf5', "r") as h5fd:
    vars_all['prec'] = h5fd['RAIN'][:][test_mask] # precipitation as always
    for _var in args.cvars:
        vars_all[_var] = h5fd[_var][:][test_mask]

ele_12km = h5py.File('../dataset/elevation_12km_resized.hdf5', "r")["elevation"]
ele_12km = np.expand_dims(np.expand_dims(ele_12km, 0), 0)
ele_12km = torch.from_numpy(ele_12km).to(torch_devs)

mask12km = np.load('../dataset/mask_12km.npy')
mask12km[mask12km > 0] = 1

for depth in (1, 8, 24, 40, 56, 72, 80, 112, )[:1]:
    if args.ckpt is None:
        # mdls = sorted(glob.glob('depth{0}-miter-itrOut/*.pth'.format(depth, )))
        mdls = sorted(glob.glob('%s-itrOut/*.pth' % (args.exp, )))
        if len(mdls) == 0: 
            print("[Error] there is not any model for depth %d" % depth)
            continue
        else:
            mdl_fn = mdls[-1]
    else:
        mdl_fn = '%s-itrOut/mdl-it%05d.pth' % (args.exp, args.ckpt)

    print("[Info] %s is used for inference" % mdl_fn)
    if args.ch1st:
        model = mdlCSR_CH(in_ch=1, ncvar=len(args.cvars), use_ele=True, sam=args.sam, cam=args.cam, \
                          stage_chs=[args.mdlsz//2**_d for _d in range(3)])
    else:
        model = mdlCSR_SP(in_ch=1, ncvar=len(args.cvars), use_ele=True, sam=args.sam, cam=args.cam, \
                          stage_chs=[args.mdlsz//2**_d for _d in range(3)])

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(mdl_fn))
    else:
        model.load_state_dict(torch.load(mdl_fn), map_location=torch.device('gpu'))
    model.to(torch_devs)
    model.eval()

    mse_outperform_cnt, cc_outperform_cnt = 0, 0
    gt, dl, bl = [], [], []
    mse_dl, mse_bl, cc_dl, cc_bl = [], [], [], []
    for idx in range(rain_12km.shape[0])[:]:
        vars4stack = []
        batch_prec= np.array([vars_all['prec'][s_idx : (s_idx+depth)] for s_idx in (idx,)], dtype=np.float32)

        for _var in args.cvars:
            batch_cvar = np.array([vars_all[_var][s_idx : (s_idx+depth)] for s_idx in (idx,)], dtype=np.float32)
            vars4stack.append(batch_cvar)

        low_res = batch_prec[0, depth-1, :, :]
        yt = rain_12km[idx+depth-1]

        batch_prec = torch.from_numpy(batch_prec).to(torch_devs)
        vars4stack = [torch.from_numpy(_cv).to(torch_devs) for _cv in vars4stack]
        with torch.no_grad():
            if args.sam:
                yp, _atten1, _atten2 = model.forward(batch_prec, vars4stack, ele_12km, ret_sam=True)
                yp = yp.cpu().numpy()[0,0,:,:]
            else:
                yp = model.forward(batch_prec, vars4stack, ele_12km).cpu().numpy()[0,0,:,:]
            if args.logtrs: yp = np.exp(yp) - 1 # transform back
            pred = np.clip(yp, a_min=0, a_max=None) * mask12km
        
        low2high = skimage.transform.resize(low_res, output_shape=(256, 512), order=3)

        gt.append(yt)
        dl.append(pred)
        bl.append(low2high)

        _cc_dl = pearsonr(yt.flatten(), pred.flatten())[0]
        cc_dl.append(_cc_dl)

        _cc_bl = pearsonr(yt.flatten(), low2high.flatten())[0]
        cc_bl.append(_cc_bl)
        if _cc_bl < _cc_dl: cc_outperform_cnt += 1

        _mse_dl = np.mean((gt[-1] - dl[-1])**2)
        mse_dl.append(_mse_dl)

        _mse_bl = np.mean((gt[-1] - bl[-1])**2)
        mse_bl.append(_mse_bl)
        if _mse_dl < _mse_bl: mse_outperform_cnt += 1

    print('[Info] %d, %d out of %d slices outperform baseline on MSE, CC when TD=%d' % (\
          mse_outperform_cnt, cc_outperform_cnt, len(gt), depth))

    print("DL-CC: 50th%%: %.3f, 75th%%: %.3f, 95th%%: %.3f" % tuple(np.percentile(cc_dl, (50, 75, 95))), )
    print("BL-CC: 50th%%: %.3f, 75th%%: %.3f, 95th%%: %.3f" % tuple(np.percentile(cc_bl, (50, 75, 95))), )

    print("DL-MSE: 50th%%: %.3f, 75th%%: %.3f, 95th%%: %.3f" % tuple(np.percentile(mse_dl, (50, 75, 95))), )
    print("BL-MSE: 50th%%: %.3f, 75th%%: %.3f, 95th%%: %.3f" % tuple(np.percentile(mse_bl, (50, 75, 95))), )
    
    with h5py.File('infer-res/pred-{}.hdf5'.format(args.exp, ), "w") as h5_fd:
        h5_fd.create_dataset("prediction", data=np.array(dl), dtype=np.float32)
        h5_fd.create_dataset("groundtruth", data=np.array(gt), dtype=np.float32)
        h5_fd.create_dataset("baseline", data=np.array(bl), dtype=np.float32)
