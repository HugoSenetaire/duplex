
import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, has_file_allowed_extension
from typing import Optional, Callable, List, Tuple, Dict, Union, cast
import numpy as np
from PIL import Image



def default_loader(path):
    return np.array(Image.open(path), dtype=np.float32)[
        ..., 0
    ]/255*2- 1  # It's a grayscale image, so we only need one channel


def find_classes(directory):
    """Finds the class folders in a dataset.
    
    Args:
        directory (string): Root directory path.
    
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    """
    classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    paired_directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    bypass_counterfactual: bool = False,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    If bypass_counterfactual is True, we will only return the path to the original image and 
    dummy counterfactuals in a random order.

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)


    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    instances_2 = []
    available_classes = set()
    for source_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[source_class]
        source_dir = os.path.join(directory, source_class)
        if not os.path.isdir(source_dir):
            print(f"Couldn't find {source_dir}")
            continue
        target_directories = {}
        for target_class in sorted(class_to_idx.keys()):
            if target_class == source_class:
                continue
            target_dir = os.path.join(paired_directory, source_class, target_class)
            if bypass_counterfactual :
                target_directories[target_class] = target_dir
            elif os.path.isdir(target_dir):
                target_directories[target_class] = target_dir
        # Target directory : For a given source, we get the corresponding target directory for each possible target source.
    
        for root, _, fnames in sorted(os.walk(source_dir, followlinks=True)):

            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                current_dic = {str(class_to_idx[source_class]): path}
                if is_valid_file(path):
                    for target_class, target_dir in target_directories.items():
                        if bypass_counterfactual:
                            item = path, "None", class_index, class_to_idx[target_class]
                            instances.append(item)
                            current_dic[str(class_to_idx[target_class])] = "None"
                        else :
                            target_path = os.path.join(target_dir, fname)        
                            if os.path.isfile(target_path) and is_valid_file(target_path):
                                item = path, target_path, class_index, class_to_idx[target_class]
                                instances.append(item)
                                current_dic[str(class_to_idx[target_class])] = target_path
                available_classes.update([source_class, target_class])
                instances_2.append((current_dic.copy(), class_index,))


    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, instances_2