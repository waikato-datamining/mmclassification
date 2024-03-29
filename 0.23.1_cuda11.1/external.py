# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) University of Waikato, Hamilton, NZ
import os
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmcv import FileClient

from .base_dataset import BaseDataset
from .builder import DATASETS


MMCLS_CLASSES = "MMCLS_CLASSES"


def find_folders(root: str,
                 file_client: FileClient) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    folders = list(
        file_client.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(root: str, folder_to_idx: Dict[str, int],
                is_valid_file: Callable, file_client: FileClient):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()

    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = file_client.join_path(root, folder_name)
        files = list(
            file_client.list_dir_or_file(
                _dir,
                list_dir=False,
                list_file=True,
                recursive=True,
            ))
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = file_client.join_path(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders


@DATASETS.register_module()
class ExternalDataset(BaseDataset):
    """External dataset for classification (based on custom.CustomDataset class).

    The dataset supports the following kinds of annotation format,
    however the second is the preferred one.

    1. An annotation file is provided, and each line indicates a sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       The annotation file (the first column is the image path and the second
       column is the index of category): ::

            folder_1/xxx.png 0
            folder_1/xxy.png 1
            folder_2/123.png 5
            folder_2/nsdf3.png 3
            ...

       Please specify the name of categories by the argument ``classes``.

    2. The samples are arranged in the specific way: ::

           data_prefix/
           ├── class_x
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           │       └── xxz.png
           └── class_y
               ├── 123.png
               ├── nsdf3.png
               ├── ...
               └── asd932_.png

    If the ``ann_file`` is specified, the dataset will be generated by the
    first way, otherwise, try the second way.

    Args:
        data_prefix (str): The path of data directory.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        ann_file (str, optional): The annotation file. If is string, read
            samples paths from the ann_file. If is None, find samples in
            ``data_prefix``. Defaults to None.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 ann_file: Optional[str] = None,
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.file_client_args = file_client_args

        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=self.load_class_labels(),
            ann_file=ann_file,
            test_mode=test_mode)

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        file_client = FileClient.infer_client(self.file_client_args,
                                              self.data_prefix)
        classes, folder_to_idx = find_folders(self.data_prefix, file_client)
        samples, empty_classes = get_samples(
            self.data_prefix,
            folder_to_idx,
            is_valid_file=self.is_valid_file,
            file_client=file_client,
        )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if self.CLASSES is not None:
            assert len(self.CLASSES) == len(classes), \
                f"The number of subfolders ({len(classes)}) doesn't match " \
                f'the number of specified classes ({len(self.CLASSES)}). ' \
                'Please check the data folder.'
        else:
            self.CLASSES = classes

        if empty_classes:
            warnings.warn(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}",
                UserWarning)

        self.folder_to_idx = folder_to_idx

        return samples

    def load_class_labels(self):
        """
        Gets the class labels from the environment variable MMCLS_CLASSES.
        Either comma-separated string of class labels or points to a file.
        If file, either contains single line with comma-separated list of class labels or one label per line.

        :return: the class names that were determined
        :rtype: list
        """
        mmseg_classes = os.getenv(MMCLS_CLASSES)
        if mmseg_classes == None:
            raise Exception("%s environment variable containing/pointing to class labels not defined!" % MMCLS_CLASSES)
        # points to file?
        if os.path.exists(mmseg_classes):
            with open(mmseg_classes, "r") as fp:
                lines = fp.readlines()
                # comma-separated or one per line?
                if len(lines) == 1:
                    result = lines[0].strip().split(",")
                else:
                    result = []
                    for line in lines:
                        line = line.strip()
                        if len(line) > 0:
                            result.append(line)
        else:
            result = mmseg_classes.split(",")
        result.sort()
        return result

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if self.ann_file is None:
            samples = self._find_samples()
        elif isinstance(self.ann_file, str):
            lines = mmcv.list_from_file(
                self.ann_file, file_client_args=self.file_client_args)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            raise TypeError('ann_file must be a str or None')

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
