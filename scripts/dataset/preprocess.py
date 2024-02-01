import pandas as pd
import PIL
import numpy as np
import os


def rescale_values(img, x_c, y_c, w, h, target_size):
    img = PIL.Image.open(img)
    height, width = np.array(img).shape[:2]
    h_scale = h / height
    w_scale = w / width
    x_c_scale = x_c / width
    y_c_scale = y_c / height
    h_updated = h_scale * target_size
    w_updated = w_scale * target_size
    x_c_updated = x_c_scale * target_size
    y_c_updated = y_c_scale * target_size
    return x_c_updated, y_c_updated, w_updated, h_updated

def helper_dataframe(files_path, box_path, lables_path, target_image_size, path_prefix):
    assert os.path.exists(files_path), 'files_path does not exist'
    assert os.path.exists(box_path), 'lables_path does not exist'
    files = np.loadtxt(files_path, dtype=str, delimiter=' ')
    bboxes = np.loadtxt(box_path, dtype=int, delimiter=' ', usecols=(0, 1,2,3,4))
    labels = np.loadtxt(lables_path, dtype=int, delimiter=' ', usecols=(1,))
    data = pd.DataFrame({
        'id' : files[:, 0].astype(np.int32),
        'path': files[:, 1],
        'x': bboxes[:, 1],
        'y': bboxes[:, 2],
        'w': bboxes[:, 3],
        'h': bboxes[:, 4],
        # Getting center co-ordinates
        'x_center' : bboxes[:, 3] / 2 + bboxes[:, 1],
        'y_center' : bboxes[:, 4] / 2 + bboxes[:, 2],
        'labels' : labels
    })
    data['path'] = data['path'].apply(lambda x: path_prefix + x)
    results = data.progress_apply(lambda x: rescale_values(x['path'], x['x_center'], x['y_center'], x['w'], x['h'], target_image_size), axis=1, result_type='expand')
    new_column_names = ['x_c_updated', 'y_c_updated', 'w_updated', 'h_updated']
    results.columns = new_column_names
    final_df = pd.concat([data, results], axis=1)
    return final_df
