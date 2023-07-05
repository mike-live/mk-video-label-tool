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

Full form:
```
python label_tool.py <video_name>
```

Simple example:
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

0. Change `data\label_names.csv` with your label classes (available only 0-9 digits as a key). Labels can be either english or russian. Use 0 as non-defined class. After the label tool is started it automatically makes copy of label names file to the same folder as video file and names it `mk_label_names_<video_name>.csv`.
1. Run label_tool with particular video.
2. Play video (<kbd>Space</kbd>) until you would like to change the class of current frame. Stop playing by <kbd>Space</kbd>.
3. Using <kbd>←</kbd> and <kbd>→</kbd> precisely position starting frame of found event.
4. Choose one of labels using keys <kbd>0-9</kbd> associated with this event.
5. Toggle state of label burning to `Burn` by pressing <kbd>t</kbd>.
6. Play video (<kbd>Space</kbd>) until event is finished. Stop playing by <kbd>Space</kbd>.
7. Using <kbd>←</kbd> and <kbd>→</kbd> precisely position finishing frame of found event.
8. Choose label <kbd>0</kbd> as non-defined class.
9. Play video (<kbd>Space</kbd>) until next event is started. Go to step 2 until the last frame or you want to interrupt (how to is explained in the next sections).
10. Find resulting label file named `mk_labels_<video_name>.csv` in the same folder as video file. It will contains only labeled frames with corresponding time in the input video timeline and assigned label.

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

### Command line params

- `--max_cache`: Cache size. Ex.: `1G`, `5M`, `100K`, `1000F`. (`G` - Gigabytes, `M` - Megabytes, `K` - Kilobytes, `F` - Max number of frames). `20F` - cache size max of 20 frames, `1G` - cache size less than 1 gigabyte, `5M` - cache size less than 5 megabytes, `100K` - cache size less than 100 kilobytes.
- `--label_names`: Path to label names file (CSV: digit,label).
- `--labels`: Path to video labels file (CSV: frame,time,label).

# Authors

Krivonosov Mikhail, Lobachevsky State University - *Implementation* - [@mike-live](https://github.com/mike-live)

# Video examples

Video example were obtained from the paper:

Sotskov, V. P., Pospelov, N. A., Plusnin, V. V., & Anokhin, K. V. (2022). Calcium Imaging Reveals Fast Tuning Dynamics of Hippocampal Place Cells and CA1 Population Activity during Free Exploration Task in Mice. International Journal of Molecular Sciences, 23(2), 638. MDPI AG. http://dx.doi.org/10.3390/ijms23020638

# License

This project is licensed under the MIT License - see the LICENSE.md file for details.