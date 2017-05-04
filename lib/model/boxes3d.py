from common import *
import numpy as np
from utils.cython_bbox import bbox_overlaps

def corners_from_box(box):
    return np.hstack((box.min(axis=0), box.max(axis=0)))


def box_from_corners(corners):
    umin,vmin,zmin,umax,vmax,zmax = corners
    box=np.array([[umin, vmin, zmin],
                  [umax, vmin, zmin],
                  [umax, vmax, zmin],
                  [umin, vmax, zmin],
                  [umin, vmin, zmax],
                  [umax, vmin, zmax],
                  [umax, vmax, zmax],
                  [umin, vmax, zmax]])

    return box

def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    xx = Yn-int((y-TOP_Y_MIN)/TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)/TOP_X_DIVISION)

    return xx, yy

def top_box_to_lidar_box(b):
    lidar_box = np.empty((b.shape[0], 8, 3), dtype=np.float32)
    for idx in range(b.shape[0]):
        lidar_box[idx, :] = box_from_corners(np.hstack((top_to_lidar_coord(b[idx,0], b[idx,1], b[idx,2]), 
                                             top_to_lidar_coord(b[idx,3], b[idx,4], b[idx,5]))))
    return lidar_box



def lidar_box_to_top_box(b):
    x0 = b[0,0]
    y0 = b[0,1]
    x1 = b[1,0]
    y1 = b[1,1]
    x2 = b[2,0]
    y2 = b[2,1]
    x3 = b[3,0]
    y3 = b[3,1]
    u0,v0=lidar_to_top_coords(x0,y0)
    u1,v1=lidar_to_top_coords(x1,y1)
    u2,v2=lidar_to_top_coords(x2,y2)
    u3,v3=lidar_to_top_coords(x3,y3)

    z0 = max(b[0,2], b[1,2], b[2,2], b[3,2]) # top
    z4 = min(b[4,2], b[5,2], b[6,2], b[7,2]) # bottom
    Zn = int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    zmax = int((z0-TOP_Z_MIN)/TOP_Z_DIVISION)
    zmin = int((z4-TOP_Z_MIN)/TOP_Z_DIVISION)

    umin=min(u0,u1,u2,u3)
    umax=max(u0,u1,u2,u3)
    vmin=min(v0,v1,v2,v3)
    vmax=max(v0,v1,v2,v3)


    # start from the top left corner and go clockwise
    top_box = box_from_corners((umin,vmin,zmin,umax,vmax,zmax))

    return top_box

def fv_projection_layer(rois):
      front_rois = np.empty((rois.shape[0], 4), dtype=np.float32)
      for idx in range(rois.shape[0]):
          box = box_from_corners(rois[idx, :])
          front_rois[idx, :] = box_to_front_proj(box)
      return front_rois

def img_projection_layer(rois, image_info):
      img_rois = np.empty((rois.shape[0], 4), dtype=np.float32)
      for idx in range(rois.shape[0]):
          box = box_from_corners(rois[idx, :])
          img_rois[idx, :] = box_to_rgb_proj(box)
      return img_rois * image_info[0, 2]

def  pred_projection_layer(pred, rois, scores, image_info):
     keep = np.where(scores[:,1] > .5)[0]
     top_corners = corner_transform_inv(top_box_to_lidar_box(rois[keep, :]), pred[keep, 24:])
     return img_projection_layer(top_corners, image_info)

def corner_transform_inv(rois_corners, pred_deltas):
    deltas = rois_corners.max(axis=1) - rois_corners.min(axis=1)
    rois_mins = rois_corners.min(axis=1)[:, np.newaxis, :]
    diagonals = np.hypot(np.hypot(deltas[:,0], deltas[:,1]), deltas[:,2])
    diagonals = diagonals[:, np.newaxis]
    pred_corners = pred_deltas.reshape((-1, 3, 8)).transpose(0, 2, 1)
    pred_corners = pred_corners + rois_mins
    pred_corners = pred_deltas * diagonals
    top_corners = np.empty_like((pred_corners.shape[0], 6))
    for idx in range(pred_corners.shape[0]):
        top_corners[idx, :] = lidar_box_to_top_box(pred_corners[idx, :])

    return top_corners



def filter_rois(rois, image_info):
  rois_rect = np.copy(img_projection_layer(rois, image_info))
  image_rect = np.array([[0, 0, image_info[0, 1], image_info[0, 0]]])
  overlaps = bbox_overlaps(
    np.ascontiguousarray(rois_rect, dtype=np.float),
    np.ascontiguousarray(image_rect, dtype=np.float))
  return np.where(overlaps.max(axis=1) > 0)[0]



def box_from_corners(corners):
    umin,vmin,zmin,umax,vmax,zmax = corners
    box=np.array([[umin, vmin, zmin],
                  [umax, vmin, zmin],
                  [umax, vmax, zmin],
                  [umin, vmax, zmin],
                  [umin, vmin, zmax],
                  [umax, vmin, zmax],
                  [umax, vmax, zmax],
                  [umin, vmax, zmax]])

    return box

def lidar_to_front_coord(xx, yy, zz):
    THETA0,THETAn = 0, int((HORIZONTAL_MAX-HORIZONTAL_MIN)/HORIZONTAL_RESOLUTION)
    PHI0, PHIn = 0, int((VERTICAL_MAX-VERTICAL_MIN)/VERTICAL_RESOLUTION)
    c = ((np.arctan2(xx, -yy) - HORIZONTAL_MIN) / HORIZONTAL_RESOLUTION).astype(np.int32)
    r = ((np.arctan2(zz, np.hypot(xx, yy)) - VERTICAL_MIN) / VERTICAL_RESOLUTION).astype(np.int32)
    yy, xx = PHIn - int(r), THETAn - int(c) 
    return xx, yy

def lidar_to_rgb_coord(xx, yy, zz):
    lidar_point = np.array([xx, yy, zz, 1.], dtype=np.float32).reshape((4, 1))
    rgb_point =  np.dot(np.array(P2), np.dot(np.array(R0_RECT), np.dot(np.array(TR_VELO_TO_CAM), lidar_point)))
    return (rgb_point[0, :] / rgb_point[2, :], rgb_point[1, :] / rgb_point[2, :])

def top_to_lidar_coord(xx, yy, zz):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    x = ((Xn - yy) - 0.5) * TOP_X_DIVISION + TOP_X_MIN
    y = ((Yn - xx) - 0.5) * TOP_Y_DIVISION + TOP_Y_MIN
    z = (zz + 0.5)*TOP_Z_DIVISION + TOP_Z_MIN
    return x,y,z

def top_to_front_coord(xx, yy, zz):
    x, y, z = top_to_lidar_coord(xx, yy, zz)
    xx, yy = lidar_to_front_coord(x, y, z)
    return xx, yy

def top_to_rgb_coord(xx, yy, zz):
    x, y, z = top_to_lidar_coord(xx, yy, zz)
    xx, yy = lidar_to_rgb_coord(x, y, z)
    return xx, yy

def box_to_front_proj(box):
    front  = np.empty([box.shape[0], 2])
    for i in range(box.shape[0]):
        front[i,:] = top_to_front_coord(*box[i,:])

    return np.hstack((front.min(axis=0), front.max(axis=0)))

def box_to_rgb_proj(box):
    rgb  = np.empty([box.shape[0], 2])
    for i in range(box.shape[0]):
        rgb[i,:] = top_to_rgb_coord(*box[i,:])

    return np.hstack((rgb.min(axis=0), rgb.max(axis=0)))
