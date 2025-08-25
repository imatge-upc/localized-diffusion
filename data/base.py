import abc
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset
from omegaconf.listconfig import ListConfig

# -- Base Dataset Class for StageIn -- #
class CustomDataset(Dataset, abc.ABC):
    def __init__(self, items, transforms):
        super().__init__()
        self.items = items
        self.transforms = transforms

    def get_labels_by_keys(self):
        """Optional. Should return all the labels given all the keys. Only needed if WeightedRandomSampler is used."""
        raise NotImplementedError

    def get_item_by_key(self, key):
        """Should return the item given a key."""
        raise NotImplementedError

    def index_to_key(self, index):
        """Optional. Use it to transform recovered item to your format so `get_item_by_key` can load the item from the key."""
        return self.items[index]

    def __getitem__(self, index):
        """Default implementation. Retrieve data and then applies transforms"""
        data = self.get_item_by_key(self.index_to_key(index))
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"{self.__class__.__name__}"


def _get_files(path: Path, pattern: str) -> List[str]:
    return [str(p.relative_to(path)) for p in path.glob(pattern) if p.is_file()]

def _get_folders(path: Path, pattern: str) -> List[str]:
    return [str(p.relative_to(path)) for p in path.glob(pattern) if p.is_dir()]

# -- Folder Datasets -- #
class FolderDataset(CustomDataset, abc.ABC):
    """Dataset that takes all the elements of a folder and builds a dataset around it"""

    ACCEPTED_FORMATS = [list, ListConfig]

    def __init__(self, items, data_root: Union[Path, str], transforms=None, file_extension=None, **kwargs):
        super().__init__(items, transforms=transforms, **kwargs)
        self.data_root = Path(data_root)
        assert self.data_root.exists(), f"data_root {self.data_root.absolute()} doesn't exist!"
        assert self.data_root.is_dir(), f"data_root {self.data_root.absolute()} isn't a directory!"
        self.file_extension = file_extension

    @staticmethod
    def format_extensions(extensions):
        if extensions:
            extensions = [extensions] if not any(
                isinstance(extensions, x) for x in FolderDataset.ACCEPTED_FORMATS 
            ) else extensions
            extensions = list(
                map(lambda e: e if "*" in e else f"*{e}", list(
                    map(lambda e: e if "." in e else f".{e}", extensions))
                )
            ) 
        else:
            extensions = []
        return extensions
    
    @staticmethod
    def format_patterns(patterns):
        if patterns:
            patterns = [patterns] if not any(
                isinstance(patterns, x) for x in FolderDataset.ACCEPTED_FORMATS
            ) else patterns
        else:
            patterns = []

        # Checking patterns
        errors = ''
        for pattern in patterns:
            try:
                flags = ['*' in p for p in [pattern[0], pattern[-1]]]
                assert all(flags)
            except Exception:
                errors += f'Pattern {pattern} needs to have "*" in front and at the end.\n'
                continue
        assert errors == '', errors
        return patterns

    @staticmethod
    def construct_patterns(patterns, extensions):
        all_patterns = []
        if not patterns:
            all_patterns += extensions
            if not all_patterns:
                all_patterns += "*"
        else:
            if extensions:
                for p in patterns:
                    for e in extensions:
                        all_patterns.append(p+e if e else p)
            else:
                all_patterns = patterns
            all_patterns = list(map(lambda p: p.replace('**', '*'), all_patterns))
        return all_patterns

    @classmethod
    def get_dataset_items(cls, data_root, patterns=None, extensions=None, recursive=True, get_files=True, sort=True):
        """Get all files in `data_root` with optional `patterns` and/or `extensions`.
        Note:
            > patterns: You should set your pattern between * to make it work (Ex: '*pattern*')
            > extensions: Yoy can set extension directly (Ex: 'extension')

        Optionally, make the search recursive (by default) and sorted (by default)
        """
        data_root = Path(data_root)

        # Define Patterns
        patterns = cls.format_patterns(patterns)

        # Define Extensions. If it's needed, add `.` to extensions. Then, add `*` to find all files. Ej: *.py
        if extensions and not get_files:
            raise ValueError("You can't set extensions if you're looking for folders.")
        extensions = cls.format_extensions(extensions)

        # Construct full patterns
        patterns = cls.construct_patterns(patterns, extensions) 
        
        # Add recurvise pattern. Ex: *.py --> **/*.py
        if recursive: 
            patterns = list(map(lambda p: f"**/{p}", patterns))

        # Find items
        items = list(map(lambda p: _get_files(data_root, p) if get_files else _get_folders(data_root, p), patterns))
        # Flatten files list
        items = [f for sublist in items for f in sublist]

        # Remove duplicates
        items = list(set(items))
        if sort: items.sort()
        return items

    def index_to_key(self, index):
        key = self.data_root / super().index_to_key(index)
        return key.with_suffix(self.file_extension) if self.file_extension else key