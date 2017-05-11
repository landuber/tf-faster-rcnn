import sys
import argparse
from xml.dom.minidom import Document
import cv2, os
import glob
import xml.etree.ElementTree as ET
import shutil
from distutils.dir_util import copy_tree
import pandas as pd
import numpy as np
import math
import shutil
import time
#from  mayavi import mlab

TOP_Y_MIN=-40  #40
TOP_Y_MAX=+40
TOP_X_MIN=-70.4
TOP_X_MAX=70.4   #70.4
TOP_Z_MIN=-1.73    ###<todo> determine the correct values!
TOP_Z_MAX=0.67

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1
TOP_Z_DIVISION=0.1


HORIZONTAL_FOV = math.pi
HORIZONTAL_MAX = HORIZONTAL_FOV
HORIZONTAL_MIN = 0.0
VERTICAL_FOV = math.pi * 41.33 / 180
VERTICAL_MAX = math.pi * 10.67 / 180
VERTICAL_MIN = -math.pi * 30.67 / 180
HORIZONTAL_RESOLUTION = HORIZONTAL_FOV / 512 #
VERTICAL_RESOLUTION = VERTICAL_FOV / 128 # 41.33 / 32

TR_VELO_TO_CAM = ([[ 0.                , -1.                ,  0.                ,  0.                ],
                   [ 0.                ,  0.                , -1.                ,  3.300000000000e-01],
                   [ 1.                ,  0.                ,  0.                ,  3.810000000000e-01],
                   [ 0.                ,  0.                ,  0.                ,  1.                ]])

R0_RECT        = ([[ 1.                ,  0.                ,  0.                , 0.                ],
                   [ 0.                ,  1.                ,  0.                , 0.                ],
                   [ 0.                ,  0.                ,  1.                , 0.                ],
                   [ 0.                ,  0.                ,  0.                , 1.                ]])

P3             = ([[ 1.362184692000e+03,  0.000000000000e+00,  6.205755310000e+02, 0.                ],
                   [ 0.000000000000e+00,  1.372305786000e+03,  5.618731330000e+02, 0.                ],
                   [ 0.000000000000e+00,  0.000000000000e+00,  1.000000000000e+00, 0.                ]])

P2             = ([[ 1.384621562000e+03,  0.000000000000e+00,  6.258880050000e+02, 0.                ],
                   [ 0.000000000000e+00,  1.393652271000e+03,  5.596263100000e+02, 0.                ],
                   [ 0.000000000000e+00,  0.000000000000e+00,  1.000000000000e+00, 0.                ]])






def generate_lidar_box_tracklet(dim, loc, rot, velo_to_cam, cam_to_velo):
  h = dim[0]
  w = dim[1]
  l = dim[2]

  # corners from the top surface, then the bottom surface
  box = np.array([ # in camera coordinates around zero point and without orientation yet\
          [l/2, l/2,  -l/2, -l/2, l/2, l/2,  -l/2, -l/2], \
          [ 0.0,  0.0,  0.0, 0.0,   -h,  -h,  -h,  -h], \
          [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2]])


  rotMat = np.array([\
          [np.cos(rot), 0.0, np.sin(rot)], \
          [        0.0,          1.0, 0.0], \
          [-np.sin(rot),0.0, np.cos(rot)]])
  loc = np.hstack((loc, [1.])).T
  loc_lidar = np.dot(velo_to_cam, loc)[:3].T
  cornerPosInCam = np.dot(rotMat, box) + np.tile(loc_lidar, (8,1)).T
  cornerPosInCam = np.vstack((cornerPosInCam, np.ones((8))))
  cornerPosInVelo = np.dot(cam_to_velo, cornerPosInCam)
  box = cornerPosInVelo[0:3, :].transpose()
  return box


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

def corners_from_box(box):
    return np.hstack((box.min(axis=0), box.max(axis=0)))

def top_corners_to_lidar_box(b):
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


def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    xx = Yn-int((y-TOP_Y_MIN)/TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)/TOP_X_DIVISION)

    return xx, yy

def generate_xml(po, camera_info,
                 class_sets = ('pedestrian', 'car', 'cyclist'), \
                 doncateothers= True):

    doc = Document()

    def append_xml_node_attr(child, parent = None, text = None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    stem = str(camera_info[0])
    img_name = stem + '.jpg'
    img_size = [camera_info[2], camera_info[1], 3]

    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent = annotation, text='KITTI')
    append_xml_node_attr('filename', parent = annotation, text=img_name)
    source = append_xml_node_attr('source', parent=annotation)
    append_xml_node_attr('database', parent=source, text='KITTI')
    append_xml_node_attr('annotation', parent=source, text='KITTI')
    append_xml_node_attr('image', parent=source, text='KITTI')
    append_xml_node_attr('flickrid', parent=source, text='000000')
    owner = append_xml_node_attr('owner', parent=annotation)
    append_xml_node_attr('url', parent=owner, text = 'http://www.cvlibs.net/datasets/kitti/index.php')
    size = append_xml_node_attr('size', annotation)
    append_xml_node_attr('width', size, str(img_size[0]))
    append_xml_node_attr('height', size, str(img_size[1]))
    append_xml_node_attr('depth', size, str(img_size[2]))
    append_xml_node_attr('segmented', parent=annotation, text='0')

    # create objects
    objs = []
    cls = po['cls'].lower()
    cls = 'dontcare' if cls not in class_sets else cls

    difficult = 0 
    truncted = 0 
    height, width, length = float(po['h']), float(po['w']), float(po['l'])
    x, y, z = float(po['tx']), float(po['ty']), float(po['tz'])

    velo_to_cam = np.array(TR_VELO_TO_CAM, dtype=np.float32)
    cam_to_velo = np.linalg.inv(velo_to_cam)
    location = np.array([x,y,z], dtype=np.float32)
    rot_cam = -math.pi/2 - float(po['rz'])
    lidar_box = generate_lidar_box_tracklet(np.array([height, width, length]), location, rot_cam, velo_to_cam,  cam_to_velo)
    corners = corners_from_box(lidar_box_to_top_box(lidar_box))
    x1_img, y1_img, x2_img, y2_img = box_to_front_proj(lidar_box)

    obj = append_xml_node_attr('object', parent=annotation)
    append_xml_node_attr('name', parent=obj, text=cls)
    append_xml_node_attr('pose', parent=obj, text='Left')
    append_xml_node_attr('truncated', parent=obj, text=str(truncted))
    append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)))
    dimensions = append_xml_node_attr('dimensions', parent=obj)
    append_xml_node_attr('height', parent=dimensions, text=str(height))
    append_xml_node_attr('width', parent=dimensions, text=str(width))
    append_xml_node_attr('length', parent=dimensions, text=str(length))

    lb = append_xml_node_attr('lidar_box', parent=obj)
    for i in range(lidar_box.shape[0]):
       cr = append_xml_node_attr('corner', parent=lb)
       append_xml_node_attr('x', parent=cr, text=str(lidar_box[i, 0]))
       append_xml_node_attr('y', parent=cr, text=str(lidar_box[i, 1]))
       append_xml_node_attr('z', parent=cr, text=str(lidar_box[i, 2]))
    location = append_xml_node_attr('location', parent=obj)
    append_xml_node_attr('x', parent=location, text=str(x))
    append_xml_node_attr('y', parent=location, text=str(y))
    append_xml_node_attr('z', parent=location, text=str(z))
                


    o = {'class': cls, 'box': lidar_box}
    objs.append(o)

    return  doc, objs


def _is_hard(cls, truncation, occlusion, x1, y1, x2, y2):
    # Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
    # Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
    # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
    hard = False
    if y2 - y1 < 25 and occlusion >= 2:
        hard = True
        return hard
    if occlusion >= 3:
        hard = True
        return hard
    if truncation > 0.8:
        hard = True
        return hard
    return hard

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert KITTI dataset into Pascal voc format')
    parser.add_argument('--kitti', dest='kitti',
                        help='path to kitti root',
                        default='./data/KITTI', type=str)
    parser.add_argument('--out', dest='outdir',
                        help='path to voc-kitti',
                        default='./data/KITTIVOC', type=str)
    parser.add_argument('--draw', dest='draw',
                        help='draw rects on images',
                        default=0, type=int)
    parser.add_argument('--dontcareothers', dest='dontcareothers',
                        help='ignore other categories, add them to dontcare rsgions',
                        default=1, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)
    args = parser.parse_args()
    return args

def build_voc_dirs(outdir):
    """
    Build voc dir structure:
        VOC2007
            |-- Annotations
                    |-- ***.xml
            |-- ImageSets
                    |-- Layout
                            |-- [test|train|trainval|val].txt
                    |-- Main
                            |-- class_[test|train|trainval|val].txt
                    |-- Segmentation
                            |-- [test|train|trainval|val].txt
            |-- JPEGImages
                    |-- ***.jpg
            |-- SegmentationClass
                    [empty]
            |-- SegmentationObject
                    [empty]
    """
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    mkdir(os.path.join(outdir, 'Lidar'))
    mkdir(os.path.join(outdir, 'SegmentationClass'))
    mkdir(os.path.join(outdir, 'SegmentationObject'))

    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'Lidar'), os.path.join(outdir, 'ImageSets', 'Main')

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 0:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    print(mlab.view())
    mlab.show(1)

def _draw_on_image(img, objs, class_sets_dict, side=0):
    colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
              (151, 0, 255), (243, 223, 48), (0, 117, 255),\
              (58, 184, 14), (86, 67, 140), (121, 82, 6),\
              (174, 29, 128), (115, 154, 81), (86, 255, 234)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ind, obj in enumerate(objs):
        if obj['box'] is None: continue
        box = lidar_box_to_top_box(obj['box'])
        #corner = corners_from_box(box)
        #corner = corner[np.newaxis, :]
        #lidar_box = top_corners_to_lidar_box(corner)
        #box = lidar_box_to_top_box(lidar_box[0, :]).astype(int)
        cls_id = class_sets_dict[obj['class']]
        rect_color = colors[cls_id % len(colors)]
        text_color = (255, 0, 255)
        if side == 0: # rgb
            x1, y1, x2, y2 = box_to_rgb_proj(box).astype(np.int32)
        elif side == 1: # top view
            x1, y1, x2, y2 = box_to_top_proj(box)
        else: # front view
            x1, y1, x2, y2 = box_to_front_proj(box)
            rect_color = (0, 0, 0)
            text_color = (0, 0, 0)


        if obj['class'] == 'dontcare':
            #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), rect_color, 1)
        text = obj['class']
        if not side == 2:
            cv2.putText(img, text, (x1-2, y2-2), font, 0.5, text_color, 1)
    return img


def lidar_to_front_coord(xx, yy, zz):
    THETA0,THETAn = 0, int((HORIZONTAL_MAX-HORIZONTAL_MIN)/HORIZONTAL_RESOLUTION)
    PHI0, PHIn = 0, int((VERTICAL_MAX-VERTICAL_MIN)/VERTICAL_RESOLUTION)
    c = ((np.absolute(np.arctan2(xx, -yy)) - HORIZONTAL_MIN) / HORIZONTAL_RESOLUTION).astype(np.int32)
    r = ((np.arctan2(zz, np.hypot(xx, yy)) - VERTICAL_MIN) / VERTICAL_RESOLUTION).astype(np.int32)
    yy, xx = PHIn - int(r), THETAn - int(c) 
    return xx, yy

def top_to_lidar_coord(xx, yy, zz):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    x = ((Xn - yy) - 0.5) * TOP_X_DIVISION + TOP_X_MIN
    y = ((Yn - xx) - 0.5) * TOP_Y_DIVISION + TOP_Y_MIN
    z = (zz + 0.5)*TOP_Z_DIVISION + TOP_Z_MIN
    return x,y,z

def lidar_to_rgb_coord(xx, yy, zz):
    lidar_point = np.array([xx, yy, zz, 1.], dtype=np.float32).reshape((4, 1))
    rgb_point =  np.dot(np.array(P2), np.dot(np.array(R0_RECT), np.dot(np.array(TR_VELO_TO_CAM), lidar_point)))
    return (rgb_point[0, :] / rgb_point[2, :], rgb_point[1, :] / rgb_point[2, :])

def top_to_front_coord(xx, yy, zz):
    x, y, z = top_to_lidar_coord(xx, yy, zz)
    xx, yy = lidar_to_front_coord(x, y, z)
    return xx, yy

def top_to_rgb_coord(xx, yy, zz):
    x, y, z = top_to_lidar_coord(xx, yy, zz)
    xx, yy = lidar_to_rgb_coord(x, y, z)
    return xx, yy
    



def box_to_top_proj(box):
    return np.hstack((box.min(axis=0)[:2], box.max(axis=0)[:2]))

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







def lidar_to_front_tensor(lidar):
    THETA0,THETAn = 0, int((HORIZONTAL_MAX-HORIZONTAL_MIN)/HORIZONTAL_RESOLUTION)
    PHI0, PHIn = 0, int((VERTICAL_MAX-VERTICAL_MIN)/VERTICAL_RESOLUTION)
    #indices = np.where((lidar[:, 0] > 0.0))[0]

    width = THETAn - THETA0
    height = PHIn - PHI0

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    cs = ((np.absolute(np.arctan2(pxs, -pys)) - HORIZONTAL_MIN) / HORIZONTAL_RESOLUTION).astype(np.int32)
    rs = ((np.arctan2(pzs, np.hypot(pxs, pys)) - VERTICAL_MIN) / VERTICAL_RESOLUTION).astype(np.int32)
    ds = np.hypot(pxs, pys)


    rcs = np.vstack((rs, cs, pzs, ds, prs)).T
    indices = np.where((rcs[:,0] < PHIn) & (rcs[:,0] >= PHI0) & (rcs[:, 1] < THETAn) & (rcs[:, 1] >= THETA0))[0]
    rcs = rcs[indices, :]
    front = np.zeros(shape=(height,width,3), dtype=np.float32)
    # Initialize with the least height
    front[:, 0] = -1.73

    for rc in rcs:
        yy, xx = -int(rc[0] - PHI0), -int(rc[1] - THETA0) 
        # rc[2] => height
        if front[yy,xx,0] < rc[2]:
            front[yy,xx, 0] = rc[2]
        # rc[3] => distance
        if front[yy,xx,1] < rc[3]:
            front[yy,xx,1] = rc[3]
        # rc[4] => intensity
        if front[yy,xx,2] < rc[4]:
            front[yy,xx,2] = rc[4]

    front[:, :, 0] = front[:, :, 0]-np.min(front[:, :, 0])
    front[:, :, 0] = (front[:, :, 0]/np.max(front[:, :, 0])*255).astype(np.uint8)
    front[:, :, 1] = front[:, :, 1]-np.min(front[:, :, 1])
    front[:, :, 1] = (front[:, :, 1]/np.max(front[:, :, 1])*255).astype(np.uint8)
    front[:, :, 2] = front[:, :, 2]-np.min(front[:, :, 2])
    front[:, :, 2] = (front[:, :, 2]/np.max(front[:, :, 2])*255).astype(np.uint8)
    front = np.dstack((front[:,:, 0], front[:,:, 1], front[:,:, 2])).astype(np.uint8)
        
    return front




def lidar_to_top_tensor(lidar):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)/TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)/TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)/TOP_Z_DIVISION).astype(np.int32)

    q_lidar = np.vstack((qxs, qys, qzs, pzs, prs)).T
    indices = np.where((q_lidar[:, 0] < Xn) & (q_lidar[:, 0] >= X0) 
                     & (q_lidar[:, 1] < Yn) & (q_lidar[:, 1] >= Y0) 
                     & (q_lidar[:, 2] < Zn) & (q_lidar[:, 2] >= Z0))[0]
    q_lidar = q_lidar[indices, :]
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    for l in q_lidar:
        yy,xx,zz = -int(l[0]-X0),-int(l[1]-Y0),int(l[2]-Z0)
        height = max(0,l[3]-TOP_Z_MIN)
        top[yy,xx,Zn+1] = top[yy,xx,Zn+1] + 1
        if top[yy, xx, zz] < height:
            top[yy,xx,zz] = height
        if top[yy, xx, Zn] < l[4]:
            top[yy,xx,Zn] = l[4]

    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)

    return top, top_image



if __name__ == '__main__':
    args = parse_args()

    _kittidir = args.kitti
    _outdir = args.outdir
    _draw = bool(args.draw)
    _dest_label_dir, _dest_img_dir, _dest_lidar_dir, _dest_set_dir = build_voc_dirs(_outdir)
    _doncateothers = bool(args.dontcareothers)

    # for kitti only provides training labels
    for dset in ['train']:

        _tracklet_path = os.path.join(_kittidir, 'tracklet_labels.xml')
        _camera_csv = os.path.join(_kittidir, 'capture_vehicle_camera.csv')
        _lidar_csv = os.path.join(_kittidir, 'capture_vehicle_lidar_interp.csv')

        """
        class_sets = ('pedestrian', 'cyclist', 'car', 'person_sitting', 'van', 'truck', 'tram', 'misc', 'dontcare')
        """
        class_sets = ('car', 'dontcare')
        class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
        allclasses = {}
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
        ftrain = open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')

        pose_objs = []
        tracklets = ET.parse(_tracklet_path).getroot().findall('tracklets')[0]
        for tracklet in tracklets.findall('item'):
            cls = tracklet.find('objectType').text
            h = tracklet.find('h').text
            w = tracklet.find('w').text
            l = tracklet.find('l').text
            poses = tracklet.find('poses')
            for pose in poses.findall('item'):
                pose_obj = {}
                pose_obj['cls'] = cls
                pose_obj['h'] = h
                pose_obj['w'] = w
                pose_obj['l'] = l
                pose_obj['tx'] = pose.find('tx').text
                pose_obj['ty'] = pose.find('ty').text
                pose_obj['tz'] = pose.find('tz').text
                pose_obj['rx'] = pose.find('rx').text
                pose_obj['ry'] = pose.find('ry').text
                pose_obj['rz'] = pose.find('rz').text
                pose_objs.append(pose_obj)
        
        camera_df = pd.read_csv(_camera_csv).values
        lidar_df = pd.read_csv(_lidar_csv).values

	indices = range(0, len(pose_objs))[::-1]
        for idx in indices[500:1000]:
	    po = pose_objs[idx]
            stem = str(camera_df[idx, 0])
            doc, objs = generate_xml(po, camera_df[idx], class_sets=class_sets, doncateothers=_doncateothers)
            img_file = os.path.join(_kittidir, camera_df[idx][4])
            img = cv2.imread(img_file)
            lidar_file = os.path.join(_kittidir, lidar_df[idx][2])
	    dest_lidar_file = os.path.join(_dest_lidar_dir, stem + '.bin')
            lidar = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))

            # Lidar 
            if 1:
                front = lidar_to_front_tensor(lidar)
                if _draw:
                    #top = _draw_on_image(top, objs, class_sets_dict)
                    front = _draw_on_image(front, objs, class_sets_dict, side=2)
                cv2.imwrite(os.path.join(_dest_img_dir, stem + '_front.png'), front)
            if 1:
                _, top = lidar_to_top_tensor(lidar)
                if _draw:
                    top = _draw_on_image(top, objs, class_sets_dict, side=1)
                cv2.imwrite(os.path.join(_dest_img_dir, stem + '_top.png'), top)
            if 1:
                if _draw:
                    top = _draw_on_image(img, objs, class_sets_dict, side=0)

            if 0:
                fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
                draw_lidar(lidar, fig=fig)
                mlab.show()

            cv2.imwrite(os.path.join(_dest_img_dir, stem + '.png'), img)
	    shutil.copyfile(lidar_file, dest_lidar_file)
            xmlfile = os.path.join(_dest_label_dir, stem + '.xml')
            with open(xmlfile, 'w') as f:
                f.write(doc.toprettyxml(indent='	'))

            ftrain.writelines(stem + '\n')

            # build [cls_train.txt]
            # Car_train.txt: 0000xxx [1 | -1]
            cls_in_image = set([o['class'] for o in objs])

            for obj in objs:
                cls = obj['class']
                allclasses[cls] = 0 \
                    if not cls in allclasses.keys() else allclasses[cls] + 1

            for cls in cls_in_image:
                if cls in class_sets:
                    fs[class_sets_dict[cls]].writelines(stem + ' 1\n')
            for cls in class_sets:
                if cls not in cls_in_image:
                    fs[class_sets_dict[cls]].writelines(stem + ' -1\n')

            if idx % 100 == 0:
                print(idx)




        (f.close() for f in fs)
        ftrain.close()

        print '~~~~~~~~~~~~~~~~~~~'
        print allclasses
        print '~~~~~~~~~~~~~~~~~~~'
        shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'val.txt'))
        shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'trainval.txt'))
        for cls in class_sets:
            shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
                            os.path.join(_dest_set_dir, cls + '_trainval.txt'))
            shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
                            os.path.join(_dest_set_dir, cls + '_val.txt'))
