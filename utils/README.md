# EGO4D Audio Visual Diarization Benchmark

The Audio-Visual Diarization (AVD) benchmark corresponds to characterizing _low-level_ information about conversational scenarios in the [EGO4D](https://ego4d-data.org/docs/) dataset.  This includes tasks focused on detection, tracking, segmentation of speakers and transcirption of speech content. To that end, we are proposing 4 tasks in this benchmark. 

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

Overall >750 hours of conversational data is provided in the first version of the AVD dataset. Out of this approximately 50 hours of data has been annotated to support these tasks. This corresponds to 572 clips. Of these 389 are training, 50 are validation and the remaining will are used for testing. 
Each clip is 5 minutes long.  The following schema summarizes some data statistics of the clips.
Speakers per clip : 4.71  
Speakers per frame : 0.74  
Speaking time in clip : 219.81 sec  
Speaking time per person in clip : 43.29 sec  
Camera wearer speaking time : 77.64 sec

Please refer to this link for detailed annotations schema. 
https://ego4d-data.org/docs/benchmarks/av-diarization/#annotation-schema

To summarise for audio annotation:
active_speaker_train.csv contains the video trackwise annotations for speech activity for all videos in the training fold.
The schema is of shape:
	Meta data (video name: clipwise person id: track number), duration of clip in video frames, video fps, array of frame wise speech activity, start frame of video track
	
df = pd.read_csv(f'{direc}/active_speaker_train.csv', names = ['meta', 'duration', 'fps', 'activity?', 'start'], delimiter = '\t')
# When opening the csv does not recognise the 'activity?' column as a list, instead it recognises it as a string hence the discrepency in duration and length of activity?.
