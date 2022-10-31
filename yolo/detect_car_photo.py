from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os



class MyHandler(FileSystemEventHandler):


    def on_any_event(self, event):
        pass

    def on_created(self, event):
        name = event.src_path[-5:-4]
        print(name)
        os.system(f'python yolov5/detect.py --img 640 --weights yolov5/best.pt --name {name} --conf-thres 0.15 --source {event.src_path}')


    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass


event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path='img', recursive=False)


observer.start()





while True:
    try:
        pass
    except KeyboardInterrupt:
        observer.stop()