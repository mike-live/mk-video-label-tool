import cv2
import numpy as np
import keyboard
import matplotlib.pyplot as plt
import time
from pathlib2 import Path
import pandas as pd
from datetime import datetime
from resize_aspect import ResizeWithAspectRatio
import textwrap
import argparse
import time
import shutil

plt.ion()

def parse_command_line():
    
    parser = argparse.ArgumentParser(
        prog='MK video label tool',
        description='Helps to mark each frame of video by digit 0-9' + 
        'representing class of this frame according to label_names.py'
    )
    parser.add_argument('input_video_file_name', metavar='Input video', type=Path,
                        help='Path to input video file')
    parser.add_argument('--labels', dest='video_labels_file_name', metavar='Video labels', type=Path,
                        default=None,
                        help='Path to video labels file (CSV: frame,time,label)')
    parser.add_argument('--cache-size', dest='cache_size', metavar='Cache size', type=str,
                        default='20F',
                        help='Cache size. Ex.: 1G, 5M, 100K, 1000F.\n' + \
                             '(G - Gigabytes, M - Megabytes, K - Kilobytes, F - Max number of frames)')
    parser.add_argument('--label_names', dest='label_names_file_name', metavar='Label names', type=Path,
                        default='data\label_names.csv',
                        help='Path to label names file (CSV: digit,label)')
    return parser.parse_args()

def logger(func):
    def make_log(*arg, **kwarg):
        
        # print('Run', func.__name__)
        be = time.time()
        val = func(*arg, **kwarg)
        en = time.time()
        # print('End', func.__name__)
        # print('Time elapsed:', en - be)
        return val
    return make_log

class VideoLabeler:
    def __init__(self, input_video_file_name, label_names_file_name, video_labels_file_name, cache_size):
        self.input_video_file_name = input_video_file_name
        if video_labels_file_name is None:
            video_labels_file_name = input_video_file_name.parent / \
                ('mk_labels_' + input_video_file_name.stem + '.csv')
        self.video_labels_file_name = video_labels_file_name
        self.label_names_file_name = self.get_label_names_file_name()
        if not self.label_names_file_name.is_file():
            if Path(label_names_file_name).is_file():
                shutil.copy(label_names_file_name, self.label_names_file_name)
            else:
                print('Please, specify file with label names (--label_names)')
                quit()
        else:
            if Path(label_names_file_name).is_file():
                print('Label names file already created for this video and it will be used:')
                print(self.label_names_file_name)
        self.load_label_names()
        
        self.cache_size = cache_size.lower()
        if self.cache_size.endswith('f'):
            self.cache_max_frames = int(self.cache_size[:-1])
        elif self.cache_size.endswith('k'):
            self.cache_max_size = int(self.cache_size[:-1]) * 10 ** 3
        elif self.cache_size.endswith('m'):
            self.cache_max_size = int(self.cache_size[:-1]) * 10 ** 6
        elif self.cache_size.endswith('g'):
            self.cache_max_size = int(self.cache_size[:-1]) * 10 ** 9

        self.is_plot = False
        
        keyboard.on_press_key("n", lambda _: self.next_frame())
        keyboard.on_press_key("p", lambda _: self.prev_frame())

        keyboard.on_press_key("right", lambda _: self.next_frame())
        keyboard.on_press_key("left", lambda _: self.prev_frame())

        keyboard.on_press_key(" ", lambda _: self.toggle_play_video())
        keyboard.on_press_key("q", lambda _: self.quit_video())
        keyboard.on_press_key("s", lambda _: self.save_labels())
        keyboard.on_press_key("t", lambda _: self.toggle_paint())

        keyboard.on_press_key("up", lambda _: self.increase_speed())
        keyboard.on_press_key("down", lambda _: self.decrease_speed())

        for i in range(0, 10):
            def f(_, p=i):
                return self.change_cur_label_video(p)
            keyboard.on_press_key(str(i), f)
        print('Trying to read video file:')
        print(str(self.input_video_file_name))
        self.cap = cv2.VideoCapture(str(self.input_video_file_name))
        ret, im = self.cap.read()
        if not ret:
            print('Video is unreadable')
            quit()
        if hasattr(self, 'cache_max_size'):
            self.cache_max_frames = max(10, int(self.cache_max_size / im.size))
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cache_max_frames)
        print('Cache max frames:', self.cache_max_frames)

        self.video_frame_id = -1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_sar = (self.cap.get(cv2.CAP_PROP_SAR_NUM), self.cap.get(cv2.CAP_PROP_SAR_DEN))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.frame_sar[0] / self.frame_sar[1])
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # * self.frame_sar[1] / self.frame_sar[0])
        print('--- Video Info ---')
        print('Frame aspect ratio:', self.frame_sar)
        print('Width and Height:', self.frame_width, self.frame_height)
        print('Num frames:', self.num_frames)
        print('FPS of video file', self.fps)
        
        self.speed = 1
        self.is_quit = False
        self.is_playing = False
        self.is_paint_label = False
        self.cur_frame = 0 
        self.labels = {}
        self.cur_label = 0
        self.last_label_t = 0
        self.prev_time_save = time.time()
        self.prev_time_backup = time.time()

    def get_label_names_file_name(self):
        return self.input_video_file_name.parent / \
            ('mk_label_names_' + self.input_video_file_name.stem + '.csv')

    def increase_speed(self):
        if self.speed < 1:
            self.speed = self.speed + 0.125
        else:
            self.speed = min(20, int(self.speed) + 1)

    def decrease_speed(self):
        if self.speed <= 1:
            self.speed = max(0.125, self.speed - 0.125)
        else:
            self.speed = max(1, int(self.speed) - 1)


    @logger
    def toggle_paint(self):
        self.is_paint_label ^= True
        self.is_plot = True

    @logger
    def paint_cur_label(self):
        if self.is_paint_label:
            self.labels[self.cur_frame] = self.cur_label
        self.save_labels(need_save=False)

    @logger
    def next_frame(self, add_frames=1):
        self.cur_frame = (self.cur_frame + add_frames) % self.num_frames
        self.is_plot = True

    @logger
    def prev_frame(self):
        self.cur_frame = (self.cur_frame - 1) % self.num_frames
        self.is_plot = True
    
    @logger
    def play_video(self):
        self.is_playing = True
    
    @logger
    def pause_video(self):
        self.is_playing = False
    
    @logger
    def toggle_play_video(self):
        self.is_playing ^= True

    @logger
    def quit_video(self):
        self.is_quit = True
        self.save_labels(need_save=True)
        quit()

    @logger
    def change_cur_label_video(self, label):
        self.cur_label = label
        self.is_plot = True
    
    @logger
    def load_label_names(self):
        try:
            df = pd.read_csv(self.label_names_file_name, index_col=None, dtype=str, keep_default_na=False)
            if not set(df['digit']).issubset(set('0123456789')):
                raise Exception('Digit column contains some non-digit values')
            if not len(set(df['digit'])) == len(df['digit']):
                raise Exception('Digit column contains duplicate values')
            self.label_names = dict(zip(df['digit'], df['label']))
        except Exception as e:
            print('Trying to read label names file, but some problems occurs:')
            print(self.label_names_file_name)
            print(repr(e))
            quit()


    @logger
    def load_labels(self):
        try:
            df = pd.read_csv(self.video_labels_file_name, index_col=None)
            self.labels = dict(zip(df['frame'], df['label']))
            if len(df):
                self.last_label_t = df['frame'].max()
            else:
                self.last_label_t = 0
        except:
            pass

    @logger
    def save_labels(self, need_save=True):
        df = pd.DataFrame(self.labels.items(), columns=['frame', 'label'])
        df['time'] = (df['frame'] / self.fps).apply('{:0.15f}'.format)
        df = df.sort_values(by=['frame'])
        if need_save or time.time() - self.prev_time_save > 5:
            df.to_csv(self.video_labels_file_name.with_suffix('.csv'), index=None)
            self.prev_time_save = time.time()

        if need_save or time.time() - self.prev_time_backup > 5 * 60:
            ts = datetime.now().strftime("%Y.%m.%d_%H%M%S")
            name = self.video_labels_file_name.stem + '_' + ts + '.csv'
            history_dir = (self.video_labels_file_name.parent / 'history')
            history_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(history_dir / name)
            self.prev_time_backup = time.time()

    @logger
    def get_frame(self):
        if 0 < self.cur_frame - self.video_frame_id < 20:
            while self.video_frame_id + 1 < self.cur_frame:
                ret, frame = self.cap.read()
                if ret:
                    self.paint_cur_label()  
                    self.video_frame_id += 1
        elif self.video_frame_id + 1 != self.cur_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame - 1)
        frame = None
        ret, frame = self.cap.read()
        if ret:
            self.paint_cur_label()
            self.video_frame_id = self.cur_frame
            return frame.copy()
        else:
            return None

    def put_multiline_text(self, frame, text, position, font, font_scale, color, thickness, line_type):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        line_height = text_size[1] + 5
        x, y = position
        for i, line in enumerate(text.split("\n")):
            cv2.putText(frame,
                        line,
                        (x, y),
                        font,
                        font_scale,
                        color,
                        thickness,
                        line_type)
            y += int(line_height * 1.7)
            
        return y
    @logger
    def plot_key_menu(self, width, height):
        key_menu = [
            ('Labels:', '')
        ]
        for key, label_name in self.label_names.items():
            key_menu.append((key, label_name))
        
        key_menu += [
            ('', ''),
            ('space', 'Play video' if not self.is_playing else 'Stop video'),
            ('t', 'Toggle burn/skip label'),
            ('Right', 'Next frame'),
            ('Left', 'Previous frame'),
            ('Up', 'Increase speed'),
            ('Down ', 'Decrease speed'),
            ('s', 'Save labels to file'),
            ('q', 'Quit'),
            ('Info:', 'Auto save: 5 sec')
        ]
        
        im = np.zeros((height, width, 3), dtype=np.uint8)
        x = 10
        y = 30
        for i, (key, message) in enumerate(key_menu):
            im = cv2.putText(
                im, key, (x, y), 
                cv2.FONT_HERSHEY_COMPLEX, 1, 
                (100, 255, 100), 2, cv2.LINE_AA, False
            )
            textsize = cv2.getTextSize(key, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
            if len(message) > 20:
                message = "\n".join(textwrap.wrap(message, 25))
            y = self.put_multiline_text(
                im, message, (x + (100 if i > len(self.label_names) + 1 else 20) + 30, y), 
                cv2.FONT_HERSHEY_COMPLEX, 1, 
                (255, 255, 255), 2, cv2.LINE_AA
            )
        
        return im

    @logger
    def plot_frame(self):
        frame = self.get_frame()
        cur_time = self.cur_frame / self.fps
        im = frame.copy()
        im = cv2.resize(im, (self.frame_width, self.frame_height))
        title = f'Frame: {self.cur_frame}. ' + \
                f'Time: {int(cur_time) // 60:02d}:{cur_time % 60:05.2f}. ' + \
                f'Speed: {self.speed}x. FPS: {self.estimated_fps:0.1f}'
        frame = cv2.putText(
            im, title, (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 255, 255), 2, cv2.LINE_AA, False
        )
        
        title = f'Label painter: {self.cur_label} '
        title += f'Label frame: {self.labels.get(self.cur_frame, -1)} | '
        title += f'{"Burn" if self.is_paint_label else "Skip"}'
        textsize = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
        
        frame = cv2.putText(
            frame, title, (frame.shape[1] - textsize[0] - 20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 255, 255), 2, cv2.LINE_AA, False
        )
        
        text = self.label_names.get(str(self.labels.get(self.cur_frame, -1)), 'None')
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]

        textX = frame.shape[1] // 2 - textsize[0] // 2
        textY = frame.shape[0] // 2 + textsize[1] // 2
        frame = cv2.putText(
            frame, text, (textX, textY), 
            cv2.FONT_HERSHEY_COMPLEX, 1, 
            (255, 0, 0), 2, cv2.LINE_AA, False
        )
        menu_im = self.plot_key_menu(width=600, height=frame.shape[0])
        frame = np.concatenate([frame, menu_im], axis=1)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        (_, _, width, height) = cv2.getWindowImageRect('Frame')
        frame1 = ResizeWithAspectRatio(frame, width=width)
        frame2 = ResizeWithAspectRatio(frame, height=height)
        frame = frame1 if frame1.shape <= frame2.shape else frame2 
        new_h, new_w = frame.shape[:2]
        img = cv2.copyMakeBorder(
            frame, 
            int((height-new_h)/2), int((height-new_h)/2), 
            int((width-new_w)/2), int((width-new_w)/2), 0
        )

        cv2.imshow('Frame', img)

    def start(self):
        self.estimated_fps = 0
        try:
            self.load_labels()
            self.cur_frame = self.last_label_t
            self.plot_frame()
            self.plot_frame()
            prv = None
            while not self.is_quit:
                if self.is_playing:
                    be = time.time()
                    self.plot_frame()
                    self.next_frame(max(1, int(self.speed)))
                    en = time.time()
                    dt_pause = max(0, 1 / self.fps / self.speed - (en - be))
                    time.sleep(dt_pause)
                    if prv is None:
                        self.estimated_fps = self.speed / (time.time() - be)
                    else:
                        k = 0.9
                        self.estimated_fps = self.estimated_fps * (1 - k) + k * self.speed / (be - prv)
                    prv = be
                elif self.is_plot:
                    self.plot_frame()
                    self.is_plot = False
                    self.estimated_fps = 0
                    prv = None
                else:
                    self.estimated_fps = 0
                    prv = None
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:    
            self.cap.release()

if __name__ == '__main__':
    args = parse_command_line()
    video_labeler = VideoLabeler(
        args.input_video_file_name, 
        args.label_names_file_name,
        args.video_labels_file_name,
        args.cache_size
    )
    video_labeler.start()