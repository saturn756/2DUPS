"""Conservative image-plane motion estimation for M4."""
from __future__ import annotations
import cv2
import numpy as np


class MotionCompensator:
    # shortcut: one similarity warp ignores parallax; extend only after depth-dependent errors are measured.
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.points = None

    def update(self, rgb, boxes):
        if min(rgb.shape[:2]) < 8:
            self.reset()
            return np.eye(2, 3), {"used": False, "reason": "gmc_small_image", "points": 0}
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None, fx=.5, fy=.5)
        height, width = gray.shape
        mask = np.full(gray.shape, 255, np.uint8)
        for box in boxes:
            padx, pady = .1 * (box.x2-box.x1), .1 * (box.y2-box.y1)
            x1,y1 = max(0,int((box.x1-padx)/2)), max(0,int((box.y1-pady)/2))
            x2,y2 = min(width,int((box.x2+padx)/2)+1), min(height,int((box.y2+pady)/2)+1)
            mask[y1:y2,x1:x2] = 0
        warp = np.eye(2, 3, dtype=np.float64)
        report = {"used":False,"reason":"gmc_first_frame","points":0}
        try:
            if self.previous is None:
                return warp, report
            if self.points is None or len(self.points) < 20:
                report["reason"] = "gmc_few_points"
                return warp, report
            forward, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None)
            if forward is None:
                report["reason"] = "gmc_flow_failed"
                return warp, report
            backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray,self.previous,forward,None)
            if backward is None:
                report["reason"] = "gmc_flow_failed"
                return warp, report
            previous, current = self.points.reshape(-1,2), forward.reshape(-1,2)
            valid = (ok.ravel()>0) & (back_ok.ravel()>0)
            valid &= np.isfinite(current).all(axis=1) & np.isfinite(backward.reshape(-1,2)).all(axis=1)
            valid &= np.linalg.norm(backward.reshape(-1,2)-previous,axis=1) < 1.5
            valid &= (current[:,0]>=0)&(current[:,0]<width)&(current[:,1]>=0)&(current[:,1]<height)
            indices = np.where(valid)[0]
            indices = indices[mask[current[indices,1].astype(int),current[indices,0].astype(int)]>0]
            previous, current = previous[indices], current[indices]
            report["points"] = len(previous)
            if len(previous) < 20:
                report["reason"] = "gmc_few_points"
                return warp, report
            estimate, inliers = cv2.estimateAffinePartial2D(previous,current,method=cv2.RANSAC,
                                                          ransacReprojThreshold=2.0)
            if estimate is None or inliers is None or not np.isfinite(estimate).all():
                report["reason"] = "gmc_fit_failed"
                return warp, report
            good = inliers.ravel().astype(bool)
            predicted = previous @ estimate[:,:2].T + estimate[:,2]
            residual = float(np.median(np.linalg.norm(predicted[good]-current[good],axis=1))) if good.any() else float("inf")
            ratio = float(good.mean())
            scale = float(np.hypot(estimate[0,0],estimate[1,0]))
            angle = float(np.degrees(np.arctan2(estimate[1,0],estimate[0,0])))
            spread = np.ptp(previous[good],axis=0) if good.any() else np.zeros(2)
            report.update(inlier_ratio=ratio,residual_px=residual*2 if np.isfinite(residual) else None,scale=scale,rotation_deg=angle)
            if (good.sum()<20 or ratio<.5 or residual>1.5 or spread[0]<.2*width or
                    spread[1]<.15*height or not .95<=scale<=1.05 or abs(angle)>5 or
                    abs(estimate[0,2])>.1*width or abs(estimate[1,2])>.1*height):
                report["reason"] = "gmc_quality_rejected"
                return warp, report
            estimate[:,2] *= 2
            report.update(used=True,reason="gmc_ok",matrix=estimate.tolist())
            return estimate, report
        except cv2.error:
            report["reason"] = "gmc_cv_error"
            return warp, report
        finally:
            self.previous = gray
            self.points = cv2.goodFeaturesToTrack(gray,maxCorners=400,qualityLevel=.01,
                                                  minDistance=6,blockSize=3,mask=mask)
