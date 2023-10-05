import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    #args.modelSavePath    = os.path.join(args.savePath, 'model')
    #args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    #args.trialPath    = os.path.join('/home/acp21jrc/Ego4D/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/', 'csv')
    #args.audioOrigPath = os.path.join(args.dataPath, 'orig_audios')
    #args.visualOrigPath= os.path.join(args.dataPath, 'orig_videos')
    #args.audioPath    = os.path.join(args.dataPath, 'wave/wave')
    #args.audioPath    = '/fastdata/acp21jrc/Ego4D/ego4d_data/ego4d_enhanced_waves'
    #args.visualPath    = '/fastdata/acp21jrc/Ego4D/ego4d_data/faces'
    args.trainTrial    = os.path.join(args.annotPath, 'active_speaker_train.csv')
    args.evalTrial     = os.path.join(args.annotPath, 'csv', f'active_speaker_{args.evalDataType}.csv')
    #args.audioPath     = os.path.join(args.dataPath, 'wave')
    #args.visualPath    = os.path.join(args.dataPath, 'video_imgs')
    #os.makedirs(args.modelSavePath, exist_ok = True)
    if args.evalDataType == 'val':
        args.evalOrig     = os.path.join(args.annotPath,'csv', 'val_orig.csv')  
    return args
 

def download_pretrain_model_AVA():
    if os.path.isfile('data/pretrain_AVA.model') == False:
        Link = "1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm"
        cmd = "gdown --id %s -O %s"%(Link, 'data/pretrain_AVA.model')
        subprocess.call(cmd, shell=True, stdout=None)
