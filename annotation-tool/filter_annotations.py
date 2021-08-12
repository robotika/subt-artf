"""Filter and convert annotations"""

import json
import os
import cv2

USED_ARTF_NAME = ["backpack", "phone", "survivor", "drill", "fire_extinguisher", "vent", "helmet", "rope", "breadcrumb",
                   "robot", "cube", "nothing"]

BLACK_LIST = [
"../virtual/helmet/local_J01_000.jpg",
"../virtual/helmet/local_J01_001.jpg",
"../virtual/helmet/local_J01_012.jpg",
"../virtual/helmet/local_J01_013.jpg",
"../virtual/helmet/local_J01_014.jpg",
"../virtual/helmet/local_J01_015.jpg",
"../virtual/helmet/local_J01_016.jpg",
"../virtual/helmet/local_J03_000.jpg",
"../virtual/helmet/local_J03_001.jpg",
"../virtual/helmet/local_J03_002.jpg",
"../virtual/helmet/local_J03_003.jpg",
"../virtual/helmet/local_J03_004.jpg",
"../virtual/helmet/local_J03_005.jpg",
"../virtual/helmet/local_J03_006.jpg",
"../virtual/helmet/local_J03_007.jpg",
"../virtual/helmet/local_J03_008.jpg",
"../virtual/helmet/local_J03_009.jpg",
"../virtual/helmet/local_J03_010.jpg",
"../virtual/helmet/local_J03_011.jpg",
"../virtual/helmet/local_J03_012.jpg",
"../virtual/helmet/local_J03_013.jpg",
"../virtual/helmet/local_J03_014.jpg",
"../virtual/helmet/local_J03_015.jpg",
"../virtual/helmet/local_J03_016.jpg",
"../virtual/helmet/local_J03_017.jpg",
"../virtual/helmet/local_J03_018.jpg",
"../virtual/drill/drill_tunel_p_013.jpg",
"../virtual/drill/drill_tunel_p_014.jpg",
"../virtual/drill/drill_tunel_p_015.jpg",
"../virtual/drill/drill_tunel_p_016.jpg",
"../virtual/drill/drill_tunel_p_039.jpg",
"../virtual/drill/drill_tunel_p_040.jpg",
"../virtual/extinguisher/ext_tunel_p_000.jpg",
"../virtual/extinguisher/ext_tunel_p_001.jpg",
"../virtual/extinguisher/ext_tunel_p_002.jpg",
"../virtual/extinguisher/ext_tunel_p_003.jpg",
"../virtual/drill/drill_tunel_s2_000.jpg",
"../virtual/drill/drill_tunel_s2_001.jpg",
"../virtual/drill/drill_tunel_s2_002.jpg",
"../virtual/drill/drill_tunel_s2_003.jpg",
"../virtual/drill/drill_tunel_s2_004.jpg",
"../virtual/extinguisher/ext_tunel_s2_000.jpg",
"../virtual/extinguisher/ext_tunel_s2_001.jpg",
"../virtual/extinguisher/ext_tunel_s2_002.jpg",
"../virtual/extinguisher/ext_tunel_s2_004.jpg",
"../virtual/backpack/fq-99-freyja-000.jpg",
"../virtual/backpack/fq-99-freyja-001.jpg",
"../virtual/backpack/fq-99-freyja-002.jpg",
"../virtual/backpack/fq-99-freyja-003.jpg",
"../virtual/backpack/fq-99-freyja-004.jpg",
"../virtual/backpack/fq-99-freyja-005.jpg",
"../virtual/backpack/fq-99-freyja-006.jpg",
"../virtual/backpack/fq-99-freyja-007.jpg",
"../virtual/backpack/fq-99-freyja-008.jpg",
"../virtual/backpack/fq-99-freyja-029.jpg",
"../virtual/backpack/fq-99-freyja-032.jpg",
"../virtual/backpack/backpack_fp3_000.jpg",
"../virtual/backpack/backpack_fp3_001.jpg",
"../virtual/backpack/backpack_fp3_002.jpg",
"../virtual/backpack/backpack_fp3_004.jpg",
"../virtual/backpack/backpack_fp3_014.jpg",
"../virtual/backpack/backpack_fp3_015.jpg",
"../virtual/helmet/helmet_fp3_001.jpg",
"../virtual/helmet/helmet_fp3_002.jpg"
]

def manual_sorting(data, annotations_dir):
    ii = 0
    while True:
        if ii < 0:
            ii = 0
        if ii >= len(data):
            ii = len(data) - 1
        file_name, artf_name, bbox, use_for = data[ii]
        x, y, xw, yh = bbox
        img = cv2.imread(os.path.join(annotations_dir, file_name), 1)
        cv2.rectangle(img, (x, y), (xw, yh), (0, 0, 255))
        cv2.putText(img, artf_name, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))
        cv2.putText(img, use_for, (10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0))
        cv2.imshow("win", img)

        k = cv2.waitKey(0) & 0xFF
        if k == ord("n"):  # next img
            ii += 1
        elif k == ord("b"):  # back one img
            ii -= 1
        elif k == ord("t"):
            data[ii][3] = "train"  # use for training
        elif k == ord("e"):
            data[ii][3] = "eval"  # use for evaluation
        elif k == ord("d"):
            data[ii][3] = "None"  # do not use
        elif k == ord("q"):  # close and save
            break

    cv2.destroyAllWindows()

    return data


def main(annotation_file, out_prefix):
    annotations_dir = os.path.dirname(annotation_file)
    data = []
    with open(annotation_file) as f:
        json_data = json.load(f)
        for item in json_data.values():
            file_name = item['filename']
            if file_name in BLACK_LIST:
                print(file_name)
                continue
            regions = item['regions']
            for reg in regions:
                artf_name = reg['region_attributes']['artifact']
                if artf_name not in USED_ARTF_NAME:
                    continue
                x = reg['shape_attributes']['x']
                y = reg['shape_attributes']['y']
                width = reg['shape_attributes']['width']
                height = reg['shape_attributes']['height']
                if g_manual:
                    # Store annotations and add a label about a future use. "None" in the beginning (do not use).
                    data.append( [file_name, artf_name, [x, y, x + width, y+ height], "None"] )
                else:
                    data.append([file_name, artf_name, [x, y, x + width, y + height], "train"])

    out_train = open(out_prefix + "_train.csv", "w")
    out_train.write("filename,class,xmin,ymin,xmax,ymax\r\n")
    if g_manual:
        data = manual_sorting(data, annotations_dir)
        out_eval = open(out_prefix + "_eval.csv", "w")
        out_eval.write("filename,class,xmin,ymin,xmax,ymax\r\n")
    for item in data:
        file_name, artf_name, bbox, use_for = item
        x, y, xw, yh = bbox
        output_string = "%s,%s,%d,%d,%d,%d\r\n" %(file_name, artf_name, x, y, xw, yh)
        if use_for == "train":
            out_train.write(output_string)
        elif use_for == "eval":
            out_eval.write(output_string)

    out_train.close()
    if g_manual:
        out_eval.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Filter and convert annotations.')
    parser.add_argument('annotation', help='json - annotations')
    parser.add_argument('--out', help='outpput csv filename', default='annotation')
    parser.add_argument('--manual', help='Manual sorting', action='store_true')
    args = parser.parse_args()
    annotation_file = args.annotation
    out_csv = args.out
    g_manual = False
    if args.manual:
        g_manual = True
    main(annotation_file, out_csv)
