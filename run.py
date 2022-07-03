import torch
import argparse
import os
import dataset as dataset
import numpy as np
import os
from models import  load_model
import multi_train as train
import torch.nn as nn
import random

def main(args):
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.BCELoss()
    
    data = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(os.getcwd(),args.data_dir)
        
    args.cuda = args.cuda and torch.cuda.is_available()
    path_model = args.model_name # in test mode model_name contains the full path

    test_model, running_params  = load_model(path_model, args) 
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        test_model = test_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    data_path = os.path.join(data, "test")

    test_dataset = dataset.MutiTaskDataset(data_path, args.seed, window_size=args.window_size, is_custom=args.custom, normalize=running_params["normalize"])


    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.cuda, collate_fn= dataset.PadCollateRaw(dim=0, predict=True),drop_last=False)


    train.test_textgrids(test_loader, test_model, criterion, device,args.window_size, args.output_dir, tv=args.tv,tc=args.tc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train creaky voice')
    parser.add_argument('--data_dir', type=str, default='allstar' , help="path to data directory",)

    parser.add_argument('--custom', action='store_true', help='is data dir a custom dataset; default is False and then AllStar is loaded;')
    parser.add_argument('--model_name', type=str, default='', help='full path of model to load')
    parser.add_argument('--output_dir', type=str, default='', help='directory to save textgrids to (leave empty if you don\'t want them to be saved)')

    parser.add_argument('--channels', type=int, default=512, metavar='N',	help='num of channels')
    parser.add_argument('--input_size', type=int, default=512, metavar='N',	help='num of features')
    parser.add_argument('--enc_hidden', type=int, default=128, metavar='N',	help='encoder intermidiate layer size')
    parser.add_argument('--hidden_size', type=int, default=256, help='classifier\'s layer size')
    parser.add_argument('--enc_out_div', type=float, default=1,	help='how much to reduct output of original encoder')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='N',	help='dropout')

    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--workers', type=int, default=20, help='number of workers to use in the data loader')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seed', type=int, default=1245,	help='random seed')
    parser.add_argument('--normalize', action='store_true', help='normalize wav')
    parser.add_argument('--window_size', type=float, default=0.005,	help='wav processing window size')
    parser.add_argument('--tv', type=float, default=0.5,	help='voicing activation function threshold')
    parser.add_argument('--tc', type=float, default=0.5,	help='pitch activation function threshold')
    args = parser.parse_args()
    main(args)