"""Causal display smoothing; never changes I5 or I6."""
import cv2
import numpy as np
from ...contracts.base import TrackState


class TrackRenderer:
    def __init__(self, alpha=1.0):
        if not 0 < alpha <= 1:
            raise ValueError("Display smoothing alpha must be within (0, 1]")
        self.alpha = alpha
        self.previous = {}

    def draw(self, rgb, tracks, frame_index):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        retained = {}
        confirmed = candidates = 0
        for track in tracks.tracks:
            if track.current_bbox is None:
                continue
            box = track.current_bbox
            current = np.array([(box.x1+box.x2)/2,(box.y1+box.y2)/2,box.x2-box.x1,box.y2-box.y1])
            previous = self.previous.get(track.track_id)
            if previous is not None:
                displacement = np.linalg.norm(current[:2]-previous[:2])
                ratio = current[2:]/previous[2:]
                if displacement < .35*np.linalg.norm(current[2:]) and np.all((ratio>.7)&(ratio<1.4)):
                    current = self.alpha*current+(1-self.alpha)*previous
            retained[track.track_id] = current
            x1,y1 = current[:2]-current[2:]/2
            x2,y2 = current[:2]+current[2:]/2
            height,width = image.shape[:2]
            x1,y1,x2,y2 = int(np.clip(x1,0,width-1)),int(np.clip(y1,0,height-1)),int(np.clip(x2,0,width-1)),int(np.clip(y2,0,height-1))
            number = int(track.track_id.rsplit(":",1)[1])
            name = track.class_ref.split(".",1)[-1].replace("_"," ")
            if track.state is TrackState.NEW:
                candidates += 1
                color, thickness, label = (170,170,170),1,f"candidate {name}"
            else:
                confirmed += 1
                color = ((number*67)%180+60,(number*97)%180+60,(number*37)%180+60)
                thickness, label = 2,f"ID {number} {name} {track.confidence:.2f}"
            cv2.rectangle(image,(x1,y1),(x2,y2),color,thickness)
            cv2.putText(image,label,(x1,max(62,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        self.previous = retained
        cv2.rectangle(image,(0,0),(image.shape[1],45),(20,20,20),-1)
        motion = "on" if tracks.association.motion_compensation_used else "off"
        text = f"YOLO11n + {tracks.association.method} | frame {frame_index} | IDs {confirmed} | candidates {candidates} | GMC {motion}"
        cv2.putText(image,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        return image
