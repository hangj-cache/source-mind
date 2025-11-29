from __future__ import print_function, division
import os
import torch
import argparse
from ADMM_Network import ESINetADMMLayer
from L21dataset import get_data_validation
import torch.utils.data as data
from scipy.io import loadmat,savemat
from os.path import join
import torch.nn as nn
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from torch.utils.tensorboard import SummaryWriter
import time


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.abs(input - target).mean()

if __name__ == '__main__':

    ###############################################################################
    # parameters----argparse一般只要一个放在main中--要用的全部参数
    ###############################################################################
    parser = argparse.ArgumentParser(description=' main ')
    # parser.add_argument('--data_dir', default='./data/training_set/data_xin_4/', type=str,
    #                     help='directory of data')
    parser.add_argument('--data_dir', default='Data', type=str,
                        help='directory of data')
    # parser.add_argument('--validation_data_dir',default='Data\\localize-mi_trans_33times\\subject01\\')
    parser.add_argument('--validation_data_dir',default='Data\Evaluation target\\')
    # parser.add_argument('--validation_data_dir',default='Data\审稿数据\\1024_validation\\')
    parser.add_argument('--brainstorm_epilepsy',default='Data\\brainstorm_ecilepsy_data')
    parser.add_argument('--three_patch',default='Data\审稿数据\\three_patch_data')
    parser.add_argument('--yokagawa',default='Data\\Yokagawa\\real_data\\')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='./logs_csnet', help='path of log files')
    parser.add_argument('--SNRs', type=str, default='10', help='signal noise ratio')
    parser.add_argument('--SNIRs', type=str, default='10', help='signal noise ratio')
    parser.add_argument('--Extents', type=str, default='5', help='area of the source')
    parser.add_argument('--patchs', type=str, default='1', help='area of the source')
    parser.add_argument('--channels',default='62',help='choose the number of channel(Data volume)')
    parser.add_argument('--cond',type=str, default='various Extents', help='Conditions for selecting research')
    parser.add_argument('--V',type=str, default='V.mat',help='Variational operator')
    parser.add_argument('--result_dir',default='./result\\DUV-L1N\\',type=str,help='the dir 0f reconstruct Source')
    args = parser.parse_args()

    ###############################################################################
    # callable methods
    ###############################################################################

    def adjust_learning_rate(opt, epo, lr):
        """Sets the learning rate to the initial LR decayed by 5 every 50 epochs"""
        lr = lr * (0.5 ** (epo // 25))  #original:50----每50个epoch调解一次学习率
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ###############################################################################
    # dataset
    ###############################################################################
    train, test, validate = get_data_validation(args.validation_data_dir,args.cond,args.Extents)
    len_train, len_test, len_validate = len(train), len(test), len(validate)
    print("len_train: ", len_train, "\tlen_test:", len_test, "\tlen_validate:", len_validate)
    train_loader = data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)   #上面num_workers说是建议在windows上设为0
    test_loader = data.DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  pin_memory=False)
    valid_loader = data.DataLoader(dataset=validate, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)

    ###############################################################################
    # mask
    ###############################################################################
    dir = join(args.validation_data_dir,args.cond)

    data = loadmat(join(dir ,args.Extents , 'L.mat'))
    mask = data['L']   #L


    mask = torch.tensor(mask, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    mask = mask.float()


    lambda1 = 10   #original ：10
    lambda2 = 1e5  #original :1e5
    delta = 0.1  #原来0.01
    ###############################################################################
    # ADMM-CSNET model
    ###############################################################################
    model = ESINetADMMLayer(mask).cuda()
    # model.reset_parameters()
    # # 统计需要更新的参数个数
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # print("Total trainable parameters:", total_params)

    num_params = count_parameters(model)
    print(f"Number of parameters:{num_params}")
    MAE = MAELoss()
    MSE = torch.nn.MSELoss(reduction='mean').cuda()

    model_params = join("logs_csnet","various conditions","0.0037918177-DUV-lam0.00001rho600000-L1N-2d-2d-600-0.001-_model_20251124_191904_78.pth")  ##

    if os.path.exists(model_params):
        params_load = torch.load(model_params)
        model.load_state_dict(params_load)

        print("==================validation======================")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (s_real, B, seedvox, TBFs, ActiveVoxSeed) in tqdm(enumerate(valid_loader), desc='valid',
                                                                          unit='file'):
                ratio = 1

                s_real = s_real.to("cuda").float().squeeze(dim=0)
                B = B.to("cuda").float().squeeze(dim=0)
                TBFs = TBFs.to("cuda").float().squeeze(dim=0)
                seedvox = seedvox.to("cuda").squeeze(dim=0)
                TBFs_tp = torch.transpose(TBFs, 0, 1)
                B_trans = torch.matmul(B,TBFs_tp)
                s_real_trans = torch.matmul(s_real,TBFs_tp)
                x = dict()
                x['B_trans'] = B_trans.unsqueeze(0)
                # start_time = time.time()
                s_gen_trans = model(x)  # 模型对象的输入是forward的输入
                # end_time = time.time()
                # print(end_time - start_time)
                s_gen_temp = s_gen_trans.squeeze(dim=0)
                s_real_temp = s_real_trans.squeeze(dim=0)
                # V_dt = V_d.t()
                s_gen = torch.matmul(s_gen_temp, TBFs)
                s_real = torch.matmul(s_real_temp, TBFs)


                filename = os.path.join(args.result_dir,args.cond,args.Extents, f'result_{batch_idx+1}.mat')

                s_gen = s_gen * ratio
                s_real = s_real * ratio
                savemat(filename,
                        {'s_reco_duvl1n': s_gen.cpu().numpy(), 's_real_duvl1n': s_real.cpu().numpy(),
                         'B_trans_duvl1n': B_trans.cpu().numpy(), 's_real_trans_duvl1n': s_real_trans.cpu().numpy(),
                         'seedvox_duvl1n': seedvox.cpu().numpy(), 'TBFs': TBFs.cpu().numpy(),'ActiveVoxSeed_duvl1n':ActiveVoxSeed.cpu().numpy()})




