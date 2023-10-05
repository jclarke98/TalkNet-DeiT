import glob, json, os
from argparse import ArgumentParser

def generate_results(split='val'):
    direc = '/home/acp21jrc/Ego4D/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/track_results/'
	
    os.makedirs('output/final_test', exist_ok=True)
    with open('data/track_results/v.txt', 'r') as f:
        videos = f.read().splitlines()
    video2res = {  v[:-4]:i for i,v in enumerate(videos)  }
    res2video = {  i:v[:-4] for i,v in enumerate(videos)  }
	
    with open(f'data/split/{split}.list', 'r') as f:
        val_videos = f.read().splitlines()
        
    print(val_videos)
    print(len(val_videos))
    tracklets = [f'{direc}{v}.txt' for v in val_videos]

    for t in tracklets[:1]:
        if not os.path.exists(t):
            print(t, 'is not in the selection')
            continue
        else:
            print(t, 'is in the selection')
        name = t[len(direc):].replace('.txt', '')
        asdres = glob.glob(f'output/results/{name}*.json')
        pidre = {}
        for asd in asdres:
            with open(asd, 'r') as f:
                lines = json.load(f)
                for line in lines:
                    identifier = '{}:{}'.format(line['frame'], line['pid'])
                    pidre[identifier] = line
        with open(t, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            data = line.split()
            identifier = '{}:{}'.format(data[0], data[1])
            if identifier in pidre:
                print('score triggered')
                print(pidre[identifier]['score'])
                print(line[:-1])
                print('*****************************************')
                new_lines.append('{} {} {}\n'.format(line[:-1], pidre[identifier]['score'], pidre[identifier]['label']))
            else:
                #print(t, line)
                new_lines.append('{} {} {}\n'.format(line[:-1], 0, 0))
        with open(f'output/final_test/{name}.txt', 'w+') as f:
            f.writelines(new_lines)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--evalDataType', type=str, default="val", help='Choose the dataset for evaluation, val or test')
    args = parser.parse_args()
    generate_results(args.evalDataType)
