import glob, json, os
from argparse import ArgumentParser

def generate_results(split='val'):
    direc = '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/'
    with open(f'v.txt', 'r') as f:
        vid_ids = f.readlines()
    vid_ids = [v.strip().split('.')[0] for v in vid_ids]
    with open(f'{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    tracklets = [f'{direc}track_results/{v}.txt' for v in videos] # accumulated tracklets per video (no ASD)

    i = 0
    for t in tracklets: # accumulated tracklets for single video (no ASD)
        if not os.path.exists(t):
            print(t, 'does not exist')
            continue
        trackid = t.split('/')[-1][:-4]
        vid_num = vid_ids.index(str(trackid))
        asdres = glob.glob(f'output/results/{trackid}*.json') # all the tracklets for a video (output of the ASD system)
        print('length of asdres', len(asdres))
        pidre = {} # asd output for a single video
        for asd in asdres: # single tracklet output of ASD system
            print(asd)
            with open(asd, 'r') as f:
                lines = json.load(f)
                for line in lines:
                    print(line)
                    identifier = '{}:{}'.format(line['frame'], line['pid'])
                    pidre[identifier] = line # {frame:pid: {frame, x1, y1, x2, y2, pid, score, label}}
        with open(t, 'r') as f: 
            lines = f.readlines() # all tracking results for a video (no ASD)
        
        new_lines = []
        for line in lines: # frame pid x1 y1 x2 y2
            line = ' '.join(line.split()[:-1]) # removes the last element (i.e. in the ground truth tracking label)
            data = line.split()
            identifier = '{}:{}'.format(data[0], data[1])
            if identifier in pidre: # if data is in ASD results (i.e. ASD has a prediction for this tracklet)
                new_lines.append('{} {} {}\n'.format(line, pidre[identifier]['score'], pidre[identifier]['label'])) 
            else:
                i += 1
                print(t, line)
                new_lines.append('{} {} {}\n'.format(line, 0, 0))
        with open(f'output/final/{vid_num}.txt', 'w+') as f:
            f.writelines(new_lines) # (frame, pid, x top left, y top left, x bottom right, y bottom right, confidence score, label)
    print('total files missing from ASD system output: ', i)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--evalDataType', type=str, default="test", help='Choose the dataset for evaluation, val or test')
    args = parser.parse_args()
    generate_results(args.evalDataType)
