import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from ASD import ASD

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=900,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    parser.add_argument('--contextLossContribution', type=float, default=0.3, help='The contribution of context loss')
    # Data path
    parser.add_argument('--savePath',     type=str, default="exps/", help='Save path of model')
    parser.add_argument('--audioPath',   type=str, default="/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/wave", help='Save path to whole audio files per clip')
    parser.add_argument('--dataPath',   type=str, default="/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/tensors/", help='Save path to tensors')
    parser.add_argument('--annotPath', default='/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/ego4d/', help='where the csv i.e active_speaker_{split}.csv is stored')
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Choose the dataset for epoch-wise evaluation, val or test')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('available device:', device)

    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    args.modelSavePath    = os.path.join(args.savePath, str(args.contextLossContribution), 'model')
    args.scoreSavePath    = os.path.join(args.savePath, str(args.contextLossContribution), 'score.txt')
    os.makedirs(args.modelSavePath, exist_ok = True)

    loader = train_loader(annotPath     = args.annotPath, \
                          audioPath     = args.audioPath, \
                          dataPath      = args.dataPath,  \
                          batchSize     = args.batchSize)
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(annotPath     = args.annotPath, \
                        audioPath     = args.audioPath, \
                        dataPath      = args.dataPath)
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
       print("Model %s loaded from previous state!"%modelfiles[-1])
       epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
       s = ASD(epoch = epoch, device = device, **vars(args))
       s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = ASD(epoch = epoch, device=device, **vars(args))
        s.loadParameters('pretrain_AVA.model')
     
    acc = []
    scoreFile = open(args.scoreSavePath, "a+")
    
    while(1):        
        loss, lossC, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            accuracy, bAP = s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
            print('here is the bappy: ', bAP)
            acc.append(accuracy)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, bestACC %2.2f%%"%(epoch, acc[-1], max(acc)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, bestACC %2.2f%%\n"%(epoch, lr, loss, acc[-1], max(acc)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()