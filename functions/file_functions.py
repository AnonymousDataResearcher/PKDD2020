"""
This module contains some commonly used functions that are not easily accessible by pathlib's Path
"""

import gzip
import pickle
import shutil
import os
from pathlib import Path


def copyfile(src, des):
    """
    Copy a file from a source to a destination

    Parameters
    ----------
    src: String
        The source filename
    des: String
        The destination filename

    Raises
    ------
    FileNotFoundError:
        If ``src`` does not exist
    """
    assert Path(src).exists()
    Path(des).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, des)


def copydir(src, des):
    assert Path(src).exists()
    Path(des).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, des)


def delete(fn):
    """
    Delete a file or folder. Does nothing if non-existent

    Parameters
    ----------
    fn: String
        The file to delete

    """
    if not Path(fn).exists():
        return
    if os.path.isdir(fn):
        shutil.rmtree(fn)
    else:
        os.remove(fn)


def list_files(fd, as_paths=True):
    """
    List the files in a directory.

    Parameters
    ----------
    fd: String
        The directory to list
    as_paths: Boolean, optional
        Return the list as absolute paths (True) or as filenames (False)

    Returns
    -------
    files: List of String
        List of paths or filenames of the files in ``fd``
    """
    assert Path(fd).exists()
    return [(os.path.join(fd, f) if as_paths else f) for f in os.listdir(fd) if
            os.path.isfile(os.path.join(fd, f))]


def list_dirs(fd, as_paths=True):
    """
    List the directories in a directory.

    Parameters
    ----------
    fd: String
        The directory to list
    as_paths: Boolean, optional
        Return the list as absolute paths (True) or as directory names (False)

    Returns
    -------
    files: List of String
        List of paths or directory names of the directories in ``fd``
    """
    assert Path(fd).exists()
    return [(os.path.join(fd, f) if as_paths else f) for f in os.listdir(fd) if
            not os.path.isfile(os.path.join(fd, f))]


def save_obj(obj, fn):
    """
    Save an object as a pickle

    Parameters
    ----------
    obj: Object
        The object to save
    fn: String
        The location to save the object

    """
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(fn):
    """
    Load an object from a pickle

    Parameters
    ----------
    fn: the location of the object

    Returns
    -------
    pkl: the object saved as pickle in ``fn``
    """
    assert Path(fn).exists()
    with open(fn, 'rb') as f:
        return pickle.load(f)


def rename(src, des):
    """
    Rename a file

    Parameters
    ----------
    src: String
        old file name
    des: String
        new file name

    Raises
    ------
    FileExistsError:
        If the new file name already exists
    """
    assert Path(src).exists()
    if Path(des).exists():
        raise FileExistsError('Destination file already exists : {}'.format(des))
    os.rename(src, des)


def extract_gz(fn, remove_old=True, new_path=None):
    """
    Extract a gz archive

    Parameters
    ----------
    fn : String
        The gz archive
    remove_old: Boolean optional
        Remove the source archive (True) or not (False)
    new_path: String or None, optional
        Destination of the unpacking. If None, the unpacking occurs in a folder with the same name as the archive in the
        same location as the archive
    """
    assert Path(fn).exists()
    assert str(fn).endswith('gz')
    if new_path is None:
        new_path = fn.replace('.gz', '')

    with gzip.open(fn, 'rb') as f_src:
        with open(new_path, 'wb') as f_des:
            shutil.copyfileobj(f_src, f_des)

    if remove_old:
        delete(fn)
