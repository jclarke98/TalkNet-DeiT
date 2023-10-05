from glob import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 
import json
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--saliencePath', type = str, default = "/mnt/parscratch/users/acp21jrc/to_visualise/1_head/salience_map",
    help = "parent directory to where csv i.e active_speaker_{split}.csv and streams for .jsons are stored")
parser.add_argument('--attentionPath', type = str, default = "/mnt/parscratch/users/acp21jrc/to_visualise/1_head/attention_map",
    help = "path to where the video images are stored")
parser.add_argument('--imagePath', type = str, default = '/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/video_imgs')
parser.add_argument('--trackPath', type = str, default = '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/infer/bbox')
parser.add_argument('--trackid', type = str, default = None)
parser.add_argument('--savePath', type = str, default = '/mnt/parscratch/users/acp21jrc/to_visualise/visualisations/1_head/')
parser.add_argument('--nSamples', type = int, default = 50)
args = parser.parse_args()

def image_loader(frame, trackid):
    video_id = trackid.split(':')[0]
    img = cv2.imread(f'{args.imagePath}/{video_id}/img_{int(frame):05d}.jpg')
    x, y = img.shape[0], img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img) 
    return img, y, x

def run(args):
    # saliency map
    salience_files = glob(f'{args.saliencePath}/*')

    # attention map
    attention_files = glob(f'{args.attentionPath}/*')

    # if args.trackid is None, randomly select trackid
    if args.trackid == None:
        print('No trackid provided, randomly selecting trackid...')
        trackids = [i.split('/')[-1].split('.')[0] for i in salience_files]
        trackid = random.choice(trackids)
    else:
        trackid = args.trackid
        
    all_tracks = glob(f'{args.trackPath}/*')
    for track in all_tracks:
        if trackid == track.split('/')[-1].split('.')[0]:
        #if trackid in track:
            print(f'found {trackid} in {track}')
            # open track
            with open(track) as f:
                bboxes = json.load(f)

    saliency_map = torch.load(f'{args.saliencePath}/{trackid}.pt', map_location="cpu")
    attention_map = torch.load(f'{args.attentionPath}/{trackid}.pt', map_location="cpu").squeeze(1)
    p_ssPath = '/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/tensors/test'
    p_ss = torch.load(f'{p_ssPath}/{trackid}/p_ss.pt', map_location="cpu") 

    # average saliency_map across last dim
    saliency_map = torch.mean(saliency_map, dim = -1)
    
    # assert first dimension of saliency_map and attention_map are equal else print error and shape
    assert saliency_map.shape[0] == attention_map.shape[0], f'saliency_map shape: {saliency_map.shape}, attention_map shape: {attention_map.shape}'

    # print shape of saliency_map and attention_map
    print('saliency_map shape: ', saliency_map.shape)
    print('attention_map shape: ', attention_map.shape)
    print('bboxes shape: ', len(bboxes))
    print('p_ss shape: ', p_ss.shape)

    # save location
    save_direc = args.savePath + '/' + trackid    
    os.makedirs(save_direc, exist_ok=True)
    
    # iterate through frames
    for i in range(0, attention_map.shape[0], round(attention_map.shape[0]/args.nSamples)):
        frame = bboxes[i]['frame']

        image, x_norm, y_norm = image_loader(frame, trackid)
        print('x_norm', x_norm)
        print('y_norm', y_norm)
        x1 = bboxes[i]['x1']/(x_norm/224)
        y1 = bboxes[i]['y1']/(y_norm/224)
        x2 = bboxes[i]['x2']/(x_norm/224)
        y2 = bboxes[i]['y2']/(y_norm/224)

        x1_pss = int(p_ss[i][0])/(x_norm/224)
        y1_pss = int(p_ss[i][1])/(y_norm/224)
        x2_pss = int(p_ss[i][2])/(x_norm/224)
        y2_pss = int(p_ss[i][3])/(y_norm/224)
        

        # assert that x1, y1, x2, y2 = x1_pss, y1_pss, x2_pss, y2_pss, else print error saying 'bboxes and p_ss do not match'
        assert x1 == x1_pss and y1 == y1_pss and x2 == x2_pss and y2 == y2_pss, 'bboxes and p_ss do not match'

        # print(bboxes[i]['x1'], int(p_ss[i][0]))
        # print(bboxes[i]['y1'], int(p_ss[i][1]))
        # print(bboxes[i]['x2'], int(p_ss[i][2]))
        # print(bboxes[i]['y2'], int(p_ss[i][3]))

        print(x1, y1, x2, y2)


        plt.figure(figsize=(20, 10))
        title = f'track id: {trackid} frame in video: {str(frame)}'
        plt.subplot(1,3,1)
        plt.title(title)
        #plt.imshow(images[i].permute(1, 2, 0).to(torch.float32))
        plt.imshow(image)
        # Create rectangle
        rect = patches.Rectangle((x1, y2), x2 - x1, y1 - y2,
                         edgecolor='green', facecolor='none', linewidth=3)
        rect_pss = patches.Rectangle((x1_pss, y2_pss), x2_pss - x1_pss, y1_pss - y2_pss,
                         edgecolor='red', facecolor='none', linewidth=5)

        # Add rectangle to axes
        plt.gca().add_patch(rect)
        plt.gca().add_patch(rect_pss)


        # reshape attention_map[i] into 14x14
        attention_map_reshaped = attention_map[i].reshape(14, 14).permute(1, 0)
        attention_map_reshaped = torch.rot90(attention_map_reshaped, -1, [0, 1])

        plt.subplot(1,3,2)
        # overlap attention_map_reshaped
        plt.imshow(attention_map_reshaped, cmap='jet')
        plt.title('softmax(QK^T) attention map')

        # reshape saliency_map[i] into 1x14x14
        saliency_map_reshaped = saliency_map[i].reshape(14, 14)
        plt.subplot(1,3,3)
        # overlay saliency_map onto image
        plt.imshow(saliency_map_reshaped, cmap = 'jet', alpha = 1)
        plt.title('Average embedding for each patch from DeiT')
        # add colorbar
        plt.colorbar()   

        plt.savefig(f'{save_direc}/{frame}.png')
        plt.close('all')
    print('*******************************************************')
    print('visualised saliency and attention maps for trackid: ', trackid)
    print('saved to: ', save_direc)
    print('*******************************************************')

if __name__ == "__main__":
    run(args)
