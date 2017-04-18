import sys
import argparse
from xml.dom.minidom import Document
import cv2, os
import glob
import xml.etree.ElementTree as ET
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import math
import shutil
import time

TOP_Y_MIN=-20  #40
TOP_Y_MAX=+20
TOP_X_MIN=0
TOP_X_MAX=40   #70.4
TOP_Z_MIN=-2    ###<todo> determine the correct values!
TOP_Z_MAX=2

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1
TOP_Z_DIVISION=0.1


#rgb camera
MATRIX_Mt = ([[  2.34773698e-04,   1.04494074e-02,   9.99945389e-01,  0.00000000e+00],
              [ -9.99944155e-01,   1.05653536e-02,   1.24365378e-04,  0.00000000e+00],
              [ -1.05634778e-02,  -9.99889574e-01,   1.04513030e-02,  0.00000000e+00],
              [  5.93721868e-02,  -7.51087914e-02,  -2.72132796e-01,  1.00000000e+00]])

MATRIX_Kt = ([[ 721.5377,    0.    ,    0.    ],
              [   0.    ,  721.5377,    0.    ],
              [ 609.5593,  172.854 ,    1.    ]])



def generate_top_box(dim, loc, rot, cam_to_velo):
  h = dim[0]
  w = dim[1]
  l = dim[2]

  box = np.array([ # in camera coordinates around zero point and without orientation yet\
          [l/2, l/2,  -l/2, -l/2, l/2, l/2,  -l/2, -l/2], \
          [ 0.0,  0.0,  0.0, 0.0,   -h,  -h,  -h,  -h], \
          [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2]])


  rotMat = np.array([\
          [np.cos(rot), 0.0, np.sin(rot)], \
          [        0.0,          1.0, 0.0], \
          [-np.sin(rot),0.0, np.cos(rot)]])
  cornerPosInCam = np.dot(rotMat, box) + np.tile(loc, (8,1)).T
  cornerPosInCam = np.vstack((cornerPosInCam, np.ones((8))))
  cornerPosInVelo = np.dot(cam_to_velo, cornerPosInCam)
  box = cornerPosInVelo[0:3, :].transpose()
  return box3d_to_top_box(box)




def box3d_to_top_box(b):
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

    z0 = min(b[0,2], b[1,2], b[2,2], b[3,2]) # top
    z4 = max(b[4,2], b[5,2], b[6,2], b[7,2]) # bottom
    Zn = int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    zmax = Zn-int((z0-TOP_Z_MIN)/TOP_Z_DIVISION)
    zmin = Zn-int((z4-TOP_Z_MIN)/TOP_Z_DIVISION)

    umin=min(u0,u1,u2,u3)
    umax=max(u0,u1,u2,u3)
    vmin=min(v0,v1,v2,v3)
    vmax=max(v0,v1,v2,v3)

    top_box=np.array([umin,vmin,zmin,umax,vmax,zmax])

    return top_box


def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    xx = Yn-int((y-TOP_Y_MIN)/TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)/TOP_X_DIVISION)

    return xx, yy

def generate_xml(name, lines, calib_lines, img_size = (370, 1224, 3), \
                 class_sets = ('pedestrian', 'car', 'cyclist'), \
                 doncateothers = True):
    """
    Write annotations into voc xml format.
    Examples:
        In: 0000001.txt
            cls        truncated    occlusion   angle   boxes                         3d annotation...
            Pedestrian 0.00         0           -0.20   712.40 143.00 810.73 307.92   1.89 0.48 1.20 1.84 1.47 8.41 0.01
        Out: 0000001.xml
            <annotation>
                <folder>VOC2007</folder>
	            <filename>000001.jpg</filename>
	            <source>
	            ...
	            <object>
                    <name>Pedestrian</name>
                    <pose>Left</pose>
                    <truncated>1</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>x1</xmin>
                        <ymin>y1</ymin>
                        <xmax>x2</xmax>
                        <ymax>y2</ymax>
                    </bndbox>
            	</object>
            </annotation>
    :param name: stem name of an image, example: 0000001
    :param lines: lines in kitti annotation txt
    :param img_size: [height, width, channle]
    :param class_sets: ('Pedestrian', 'Car', 'Cyclist')
    :return:
    """

    doc = Document()

    def append_xml_node_attr(child, parent = None, text = None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    img_name = name+'.jpg'

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
    append_xml_node_attr('width', size, str(img_size[1]))
    append_xml_node_attr('height', size, str(img_size[0]))
    append_xml_node_attr('depth', size, str(img_size[2]))
    append_xml_node_attr('segmented', parent=annotation, text='0')

    # create objects
    objs = []
    for line in lines:
        splitted_line = line.strip().lower().split()
        cls = splitted_line[0].lower()
        if not doncateothers and cls not in class_sets:
            continue
        cls = 'dontcare' if cls not in class_sets else cls
        obj = append_xml_node_attr('object', parent=annotation)

        truncation = float(splitted_line[1])
        occlusion = int(float(splitted_line[2]))
        alpha = float(splitted_line[3])
        x1, y1, x2, y2 = int(float(splitted_line[4]) + 1), int(float(splitted_line[5]) + 1), \
                         int(float(splitted_line[6]) + 1), int(float(splitted_line[7]) + 1)
        difficult = 1 if _is_hard(cls, truncation, occlusion, x1, y1, x2, y2) else 0
        truncted = 0 if truncation < 0.5 else 1
        height, width, length = float(splitted_line[8]), float(splitted_line[9]), float(splitted_line[10])
        x, y, z = float(splitted_line[11]), float(splitted_line[12]), float(splitted_line[13])
        rot_y = float(splitted_line[14])

        append_xml_node_attr('name', parent=obj, text=cls)
        append_xml_node_attr('pose', parent=obj, text='Left')
        append_xml_node_attr('truncated', parent=obj, text=str(truncted))
        append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)))
        bb = append_xml_node_attr('bndbox', parent=obj)
        append_xml_node_attr('xmin', parent=bb, text=str(x1))
        append_xml_node_attr('ymin', parent=bb, text=str(y1))
        append_xml_node_attr('xmax', parent=bb, text=str(x2))
        append_xml_node_attr('ymax', parent=bb, text=str(y2))
        dimensions = append_xml_node_attr('dimensions', parent=obj)
        append_xml_node_attr('height', parent=dimensions, text=str(height))
        append_xml_node_attr('width', parent=dimensions, text=str(width))
        append_xml_node_attr('length', parent=dimensions, text=str(length))

        velo_to_cam = np.array(calib_lines[5].strip().lower().split()[1:], dtype=np.float32)
        velo_to_cam = velo_to_cam.reshape((-1, 4))
        velo_to_cam = np.vstack((velo_to_cam, [0., 0., 0., 1.]))
        cam_to_velo = np.linalg.inv(velo_to_cam)
        location = np.array([x,y,z], dtype=np.float32)
        top_box = generate_top_box(np.array([height, width, length]), location, rot_y, cam_to_velo)
        tb = append_xml_node_attr('topbox', parent=obj)
        append_xml_node_attr('xmin', parent=tb, text=str(top_box[0]))
        append_xml_node_attr('ymin', parent=tb, text=str(top_box[1]))
        append_xml_node_attr('zmin', parent=tb, text=str(top_box[2]))
        append_xml_node_attr('xmax', parent=tb, text=str(top_box[3]))
        append_xml_node_attr('ymax', parent=tb, text=str(top_box[4]))
        append_xml_node_attr('zmax', parent=tb, text=str(top_box[5]))
        location = append_xml_node_attr('location', parent=obj)
        append_xml_node_attr('x', parent=location, text=str(x))
        append_xml_node_attr('y', parent=location, text=str(y))
        append_xml_node_attr('z', parent=location, text=str(z))
        append_xml_node_attr('rotation_y', parent=obj, text=str(splitted_line[14]))
            


        o = {'class': cls, 'box': top_box, \
             'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
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

def _draw_on_image(img, objs, class_sets_dict):
    colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
              (151, 0, 255), (243, 223, 48), (0, 117, 255),\
              (58, 184, 14), (86, 67, 140), (121, 82, 6),\
              (174, 29, 128), (115, 154, 81), (86, 255, 234)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ind, obj in enumerate(objs):
        if obj['box'] is None: continue
        x1, y1, x2, y2 = obj['box'].astype(int)
        cls_id = class_sets_dict[obj['class']]
        print('drawing' + obj['class'])
        if obj['class'] == 'dontcare':
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls_id % len(colors)], 1)
        text = '{:s}*|'.format(obj['class'][:3]) if obj['difficult'] == 1 else '{:s}|'.format(obj['class'][:3])
        text += '{:.1f}|'.format(obj['truncation'])
        text += str(obj['occlusion'])
        cv2.putText(img, text, (x1-2, y2-2), font, 0.5, (255, 0, 255), 1)
    return img


def lidar_to_tensor(lidar):
    start = time.time()
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
    indices = np.where((q_lidar[:,0] < Xn) & (q_lidar[:,0] >= X0) & (q_lidar[:, 1] < Yn) & (q_lidar[:, 1] >= Y0) & (q_lidar[:,2] < Zn) & (q_lidar[:,2] >= Z0))[0]
    q_lidar = q_lidar[indices, :]
    print('height,width,channel=%d,%d,%d'%(height,width,channel))
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
    end = time.time()
    print(end-start)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)

    return top, top_image


def lidar_to_top(lidar):

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

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    ## start to make top  here !!!
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0


                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top[yy,xx,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top[yy,xx,Zn+1]+=count

                pass
            pass
        pass
    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 0: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)


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

        _labeldir = os.path.join(_kittidir, 'training', 'label_2')
        _imagedir = os.path.join(_kittidir, 'training', 'image_2')
        _lidardir = os.path.join(_kittidir, 'training', 'velodyne')
        _calibdir = os.path .join(_kittidir, 'training', 'calib')
        """
        class_sets = ('pedestrian', 'cyclist', 'car', 'person_sitting', 'van', 'truck', 'tram', 'misc', 'dontcare')
        """
        #copy_tree(_lidardir, _dest_lidar_dir, update=1)
        class_sets = ('pedestrian', 'cyclist', 'car', 'dontcare')
        class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
        allclasses = {}
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
        ftrain = open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')

        files = glob.glob(os.path.join(_labeldir, '*.txt'))
        files.sort()
        for file in files[:]:
            path, basename = os.path.split(file)
            stem, ext = os.path.splitext(basename)
            with open(file, 'r') as f:
                lines = f.readlines()
            calib_file = os.path.join(_calibdir, stem + '.txt')
            with open(calib_file, 'r') as cf:
                calib_lines = cf.readlines()
            img_file = os.path.join(_imagedir, stem + '.png')
            img = cv2.imread(img_file)
            img_size = img.shape
            doc, objs = generate_xml(stem, lines, calib_lines, img_size, class_sets=class_sets, doncateothers=_doncateothers)

            # Lidar related code
            src_lidar_file = os.path.join(_lidardir, stem + '.bin')
            lidar_file = os.path.join(_dest_lidar_dir, stem + '.bin')
            shutil.copy2(src_lidar_file, lidar_file)
            lidar2_file = os.path.join(_dest_lidar_dir, stem + '.npy')
            lidar3_file = os.path.join(_dest_lidar_dir, stem + '.png')
            lidar = np.fromfile(lidar_file, dtype=np.float32)
            lidar = lidar.reshape((-1, 4))
            print('discretizing started')
            #lidar, top = lidar_to_top(lidar)
            lidar, top = lidar_to_tensor(lidar)
            print('discretizing ended')
            np.save(lidar2_file,lidar)

            if _draw:
                top = _draw_on_image(top, objs, class_sets_dict)

            cv2.imwrite(lidar3_file,top)
            cv2.imwrite(os.path.join(_dest_img_dir, stem + '.jpg'), img)
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

            if int(stem) % 100 == 0:
                print(file)

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
