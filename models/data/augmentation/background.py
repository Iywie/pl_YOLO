import numpy as np


def getBackground(images, labels, class_list):

    bg = []  # background blocks
    obj = []
    bg_c = []  # background blocks with class
    obj_c = []  # objects with class
    for i in range(len(class_list)):
        bg_c.append([])
        obj_c.append([])

    for image, label in zip(images, labels):
        if len(label[0]) == 0:
            continue
        xmin = int(label[0][:, 0].min())
        ymin = int(label[0][:, 1].min())
        xmax = int(label[0][:, 2].max())
        ymax = int(label[0][:, 3].max())
        (h, w, c) = image.shape[:3]

        for res in label[0]:
            cls = int(res[4])
            obj.append(image[int(res[1]):int(res[3]), int(res[0]):int(res[2])])
            obj_c[cls].append(image[int(res[1]):int(res[3]), int(res[0]):int(res[2])])

        # extract edge non-defect area to background-class list
        clss = label[0][:, 4]
        clss = np.unique(clss)
        for cls in clss:
            cls = int(cls)
            if xmin > 10 and ymin > 10:
                bg_c[cls].append(image[:ymin, :xmin])
                bg.append(image[:ymin, :xmin])
            if w - xmax > 10 and h - ymax > 10:
                bg_c[cls].append(image[ymax:h, xmax:w])
                bg.append(image[ymax:h, xmax:w])
            if xmin > 10 and h - ymax > 10:
                bg_c[cls].append(image[ymax:h, :xmin])
                bg.append(image[ymax:h, :xmin])
            if w - xmax > 10 and ymin > 10:
                bg_c[cls].append(image[:ymin, xmax:w])
                bg.append(image[:ymin, xmax:w])

    return bg, bg_c, obj_c
