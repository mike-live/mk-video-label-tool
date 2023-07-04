# MK Video Label Tool

This tool helps to label frames in video by classes from the pre-defined list.
Input: video
Output: CSV table with labels for labeled frames

### Pre-requisites

Python 3
```
pip install -r requirements.txt
```

### Run

```
python label_tool.py data\movie.mp4 
```

### Menu

Available keys:
- <kbd>Space</kbd>: Play/Stop video
- <kbd>t</kbd>: Change burn label state
- <kbd>→</kbd>: Next frame
- <kbd>←</kbd>: Previous frame
- <kbd>q</kbd>: Quit
- <kbd>s</kbd>: Save current labels to file

To choose class use digits <kbd>0-9</kbd>.

## How to use

Here we show 3 main ways of using the MK Video Label Tool: labeling, viewing, interruption of process and continuing. 

### General labeling procedure

0. Change label_names.py with your label classes (available only 0-9 digits as a key). Labels can be either english or russian. Use 0 as non-defined class.
1. Run label_tool with particular video.
2. Play video (<kbd>Space</kbd>) until you would like to change the class of current frame. Stop playing by <kbd>Space</kbd>.
3. Using <kbd>←</kbd> and <kbd>→</kbd> precisely position starting frame of found event.
4. Choose one of labels using keys <kbd>0-9</kbd> associated with this event.
5. Toggle state of label burning to `Burn` by pressing <kbd>t</kbd>.
6. Play video (<kbd>Space</kbd>) until event is finished. Stop playing by <kbd>Space</kbd>.
7. Using <kbd>←</kbd> and <kbd>→</kbd> precisely position finishing frame of found event.
8. Choose label <kbd>0</kbd> as non-defined class.
9. Play video (<kbd>Space</kbd>) until next event is started. Go to step 2 until the last frame or you want to interrupt (how to is explained in the next sections).

Caution: After the last frame, video is restarting from the beggining automatically.  

### Saving labels

By default labels are saved in the same folder where video placed. Labels are saved automatically each 5 seconds and backuped each 5 minutes in subfolder `history`. In case you would like to save label manually press <kbd>s</kbd> at any moment.

In case you would like to choose folder and file name of the `label file` use key in command line `--labels`. Example:
```
python label_tool.py data\movie.mp4 --labels data\my_labels.csv
```

### Interrupting and continuing

You can stop labeling process at any time and quit program by pressing <kbd>q</kbd>.
Labels are automatically saved to the file on quit. 

To continue labeling process start label tool with the same input video file (and in case you manually chosen `label file` specify its too). By default program restarts from the last frame in the `label file`.

### Viewing process

After labeling finished you can view video and check labels: 

1. Run label_tool with particular video.
2. Play video (<kbd>Space</kbd>) and pause at any time.
3. In the center of the frame the label name of the current frame is written.

# Authors

Krivonosov Mikhail, Lobachevsky State University - *Implementation* - @mike-live

# License

This project is licensed under the MIT License - see the LICENSE.md file for details.