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
from label_names import label_names
import time

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
    
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    return parser.parse_args()

def logger(func):
    def make_log(*arg, **kwarg):
        #print('Run', func.__name__)
        val = func(*arg, **kwarg)
        #print('End', func.__name__)
        return val
    return make_log

class VideoLabeler:
    def __init__(self, input_video_file_name, label_names, video_labels_file_name):
        self.input_video_file_name = input_video_file_name
        if video_labels_file_name is None:
            video_labels_file_name = input_video_file_name.parent / \
                ('mk_labels_' + input_video_file_name.stem + '.csv')
        self.video_labels_file_name = video_labels_file_name
        self.label_names = label_names
        
        keyboard.on_press_key("right", lambda _: self.next_frame())
        keyboard.on_press_key("left", lambda _: self.prev_frame())
        keyboard.on_press_key(" ", lambda _: self.toggle_play_video())
        keyboard.on_press_key("q", lambda _: self.quit_video())
        keyboard.on_press_key("s", lambda _: self.save_labels())
        keyboard.on_press_key("t", lambda _: self.toggle_paint())
        for i in range(0, 10):
            def f(_, p=i):
                return self.change_cur_label_video(p)
            keyboard.on_press_key(str(i), f)
        print(str(self.input_video_file_name))
        self.cap = cv2.VideoCapture(str(self.input_video_file_name))
        self.frame_cache = {}
        self.video_frame_id = -1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.num_frames)
        print(self.fps)
        print(self.cap)
        print(self.cap.isOpened())

        self.is_quit = False
        self.is_playing = False
        self.is_paint_label = False
        self.cur_frame = 0 
        self.labels = {}
        self.cur_label = 0
        self.last_label_t = 0
        self.prev_time_save = time.time()
        self.prev_time_backup = time.time()

    @logger
    def toggle_paint(self):
        self.is_paint_label ^= True
        self.plot_frame()

    @logger
    def paint_cur_label(self):
        if self.is_paint_label:
            self.labels[self.cur_frame] = self.cur_label
        self.save_labels(need_save=False)

    @logger
    def next_frame(self):
        self.cur_frame = (self.cur_frame + 1) % self.num_frames
        self.plot_frame()

    @logger
    def prev_frame(self):
        self.cur_frame = (self.cur_frame - 1) % self.num_frames
        self.plot_frame()
    
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
        self.plot_frame()
    
    @logger
    def load_labels(self):
        try:
            df = pd.read_csv(self.video_labels_file_name, index_col=None)
            self.labels = dict(zip(df['frame'], df['label']))
            #print(self.labels)
            self.last_label_t = df['frame'].max()
        except:
            pass

    @logger
    def save_labels(self, need_save=True):
        df = pd.DataFrame(self.labels.items(), columns=['frame', 'label'])
        df['time'] = (df['frame'] / self.fps).apply('{:0.2f}'.format)
        
        if need_save or time.time() - self.prev_time_save > 5:
            print(self.video_labels_file_name)
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
        if self.cur_frame in self.frame_cache:
            return self.frame_cache[self.cur_frame]
        self.video_frame_id = self.cur_frame
        while not self.video_frame_id in self.frame_cache and self.cur_frame - self.video_frame_id < 20:
            self.video_frame_id -= 1
        self.video_frame_id = max(-1, self.video_frame_id)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_id)
        while self.video_frame_id < self.cur_frame:
            ret, frame = self.cap.read()
            if ret:
                self.video_frame_id += 1
                self.frame_cache[self.video_frame_id] = frame
            else:
                break
        if self.cur_frame in self.frame_cache:
            return self.frame_cache[self.cur_frame]
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
            y += line_height * 2
            
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
            ('t', 'Change burn label state'),
            ('→', 'Next frame'),
            ('←', 'Previous frame'),
            ('q', 'Quit'),
            ('s', 'Save current labels to file'),
            ('Info:', 'Labels are saved\nautomatically each 5 sec\nand backuped each 5 min')
        ]
        
        im = np.zeros((height, width, 3), dtype=np.uint8)
        x = 10
        y = 40
        for key, message in key_menu:
            #text = f'{key}: {message}'
            im = cv2.putText(
                im, key, (x, y), 
                cv2.FONT_HERSHEY_COMPLEX, 1, 
                (100, 255, 100), 2, cv2.LINE_AA, False
            )
            textsize = cv2.getTextSize(key, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
            if len(message) > 20:
                message = "\n".join(textwrap.wrap(message, 20))
            y = self.put_multiline_text(
                im, message, (x + textsize[0] + 30, y), 
                cv2.FONT_HERSHEY_COMPLEX, 1, 
                (255, 255, 255), 2, cv2.LINE_AA
            )
        
            # im = cv2.putText(
            #     im, message, (x + textsize[0] + 30, y), 
            #     cv2.FONT_HERSHEY_COMPLEX, 1, 
            #     (255, 255, 255), 2, cv2.LINE_AA, False
            # )
            
        return im

    @logger
    def plot_frame(self):
        self.paint_cur_label()
        frame = self.get_frame()
        # if IM is None:
        #     IM = ax.imshow(frame)
        # else:
        #     IM.set_data(frame)
        cur_time = self.cur_frame / self.fps
        
        im = frame.copy()
        title = f'Frame: {self.cur_frame}. ' + \
                f'Time: {int(cur_time) // 60:02d}:{cur_time % 60:05.2f}'
        frame = cv2.putText(
            im, title, (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 0, 0), 2, cv2.LINE_AA, False
        )
        
        title = f'Label painter: {self.cur_label} '
        title += f'Label frame: {self.labels.get(self.cur_frame, -1)} | '
        title += f'{"Burn" if self.is_paint_label else "Skip"}'
        
        frame = cv2.putText(
            frame, title, (1250, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 0, 0), 2, cv2.LINE_AA, False
        )
        
        text = self.label_names.get(str(self.labels.get(self.cur_frame, -1)), 'None')
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]

        textX = 975 - textsize[0] // 2
        textY = 875 + textsize[1] // 2
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
        try:
            self.load_labels()
            self.cur_frame = self.last_label_t
            self.plot_frame()
            self.plot_frame()
            while not self.is_quit:
                if self.is_playing:
                    be = time.time()
                    self.plot_frame()
                    self.next_frame()
                    en = time.time()
                    dt_pause = max(0, 1 / self.fps - (en - be))
                    time.sleep(dt_pause)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:    
            self.cap.release()

if __name__ == '__main__':
    args = parse_command_line()
    video_labeler = VideoLabeler(
        args.input_video_file_name, 
        label_names,
        args.video_labels_file_name
    )
    video_labeler.start()