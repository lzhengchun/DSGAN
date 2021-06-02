import numpy as np 
from util import *
import sys, os, time, argparse, shutil, h5py, torch
from scipy.stats import pearsonr
import multiprocessing as mp
from models import discModel
from models import encodedGenerator
from data import bkgdGen, gen_train_batch_bg, get1batch4test

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-depth',  type=int, default=1, help='input depth')
parser.add_argument('-lr',     type=float, default=3e-4, help='learning rate')
parser.add_argument('-maxep',  type=int, default=8000, help='max training epoches')
parser.add_argument('-warmup', type=int, default=100, help='warmup training epoches')
parser.add_argument('-mbsize', type=int, default=64, help='mini batch size')
parser.add_argument('-mdlsz',  type=int, default=256, help='channels of the 1st box')
parser.add_argument('-print',  type=str2bool, default=False, help='1:print to terminal; 0: redirect to file')
parser.add_argument('-logtrs', type=str2bool, default=False, help='log transform')
parser.add_argument('-sam',    type=str2bool, default=True, help='apply spatial attention')
parser.add_argument('-cam',    type=str2bool, default=True, help='apply channel attention')
parser.add_argument('-cvars',  type=str2list, default='T2:IWV:SLP', help='vars as condition')
parser.add_argument('-wmse',   type=float, default=5, help='weight of content loss for G loss')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("log to %s, log Trans: %s, mb: %d, mdlsz: %d, CAM: %s, SAM: %s, cvars:%s, wmse:%.1f" % (\
     'Terminal' if args.print else 'file', args.logtrs, args.mbsize, args.mdlsz,\
     args.cam, args.sam, ','.join(args.cvars), args.wmse))

itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

# redirect print to a file
if args.print == 0:
    sys.stdout = open(os.path.join(itr_out_dir, 'iter-prints.log'), 'w') 

def main(args):
    mb_size = args.mbsize
    in_depth = args.depth

    gene_model = encodedGenerator(in_ch=1, ncvar=len(args.cvars), use_ele=True, sam=args.sam, cam=args.cam, \
                                  stage_chs=[args.mdlsz//2**_d for _d in range(3)])

    disc_model = discModel()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            gene_model = torch.nn.DataParallel(gene_model)
            disc_model = torch.nn.DataParallel(disc_model)
        gene_model = gene_model.to(torch_devs)
        disc_model = disc_model.to(torch_devs)
    # gene_criterion = torch.nn.MSELoss()
    gene_criterion = torch.nn.L1Loss()
    disc_criterion = torch.nn.BCELoss()

    gene_optimizer = torch.optim.Adam(gene_model.parameters(), lr=args.lr)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.lr)

    lrdecay_lambda = lambda epoch: cosine_decay(epoch, warmup=args.warmup, max_epoch=args.maxep)
    # initial lr times a given function
    gene_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gene_optimizer, lr_lambda=lrdecay_lambda)
    disc_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lrdecay_lambda)
    # build minibatch data generator with prefetch
    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(mb_size=mb_size, in_depth=in_depth, \
                                                             dev=torch_devs, trans=args.logtrs, cvars=args.cvars), \
                           max_prefetch=mb_size*4)

    ele_12km = h5py.File('../dataset/elevation_12km_resized.hdf5', "r")["elevation"]
    ele_12km = np.array([ele_12km] * mb_size)
    ele_12km = np.expand_dims(ele_12km, 1)
    ele_12km = torch.from_numpy(ele_12km).to(torch_devs)

    # get disc out size and create label
    dsc_out_size= (mb_size, 1, 4, 8)
    true_label  = torch.ones (dsc_out_size)
    false_label = torch.zeros(dsc_out_size)
    disc_label  = torch.cat((true_label, false_label), dim=0).to(torch_devs)

    for epoch in range(args.maxep+1):
        time_it_st = time.time()
        X_mb, cvars, y_mb = mb_data_iter.next() # with prefetch

        # generator optimize
        gene_optimizer.zero_grad()
        pred = gene_model.forward(X_mb, cvars, ele_12km)
        with torch.no_grad():
            advs_loss = 0 - disc_model.forward(pred).mean().log() # adv loss
        cont_loss = gene_criterion(pred, y_mb) # content loss
        gene_loss = args.wmse * cont_loss + advs_loss
        gene_loss.backward()
        gene_optimizer.step() 
        gene_lr_scheduler.step()

        # discriminator optimize
        disc_optimizer.zero_grad()
        disc_mb   = torch.cat((y_mb, pred.detach()), dim=0)
        disc_pred = disc_model.forward(disc_mb)
        disc_loss = disc_criterion(disc_pred, disc_label) / 2 # slows down the rate relative to G
        disc_loss.backward()
        disc_optimizer.step()
        disc_lr_scheduler.step()

        itr_prints = '[Info] @ %.1f Epoch: %05d, gloss: %.2f = (Cont:%.2f + Adv:%.2f), dloss: %.2f, elapse: %.2fs/itr, lr: %.5f' % (\
                     time.time(), epoch, gene_loss.detach().cpu().numpy(), cont_loss.detach().cpu().numpy(), \
                     advs_loss.detach().cpu().numpy(), disc_loss.detach().cpu().numpy(), (time.time() - time_it_st), \
                     gene_optimizer.param_groups[0]['lr'])
        print(itr_prints)

        if epoch % (500) == 0:
            if epoch == 0: 
                X222, cv222, y222 = get1batch4test(in_depth=in_depth, idx=range(args.mbsize), dev=torch_devs, \
                                            trans=args.logtrs, cvars=args.cvars)
                save2img_rgb(X222[0,in_depth-1,:,:].cpu(), '%s/low-res.png' % (itr_out_dir))
                true_img = y222.cpu().numpy()
                if args.logtrs: true_img = np.exp(true_img) - 1 # transform back
                save2img_rgb(true_img[0,0,:,:], '%s/high-res.png' % (itr_out_dir))

            with torch.no_grad():
                pred_img = gene_model.forward(X222, cv222, ele_12km).cpu().numpy()
                if args.logtrs: pred_img = np.exp(pred_img) - 1 # transform back
                mse = np.mean((true_img - pred_img)**2)
                cc_avg = np.mean([pearsonr(pred_img[i].flatten(), true_img[i].flatten())[0] for i in range(pred_img.shape[0])])
            print('[Validation] @ Epoch: %05d MSE: %.4f, CC:%.3f of %d samples' % (epoch, mse, cc_avg, pred_img.shape[0]))

            save2img_rgb(pred_img[0,0,:,:], '%s/it%05d.png' % (itr_out_dir, epoch))

            if torch.cuda.device_count() > 1:
                torch.save(gene_model.module.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
            else:
                torch.save(gene_model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))

        sys.stdout.flush()

if __name__ == '__main__':
    main(args)
