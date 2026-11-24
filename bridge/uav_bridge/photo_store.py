"""Private, bounded, durable original RGB photographs; never accept file paths over HTTP."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import uuid

_RUNTIME = Path(__file__).resolve().parents[2] / '.runtime'
_ID = re.compile(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}')


class PhotoStore:
    def __init__(self, config=None):
        config = config or {}
        self.directory = Path(config.get('store_directory') or str(_RUNTIME / 'photos')).absolute()
        # No arbitrary files, symlink destinations, or external directory supplied
        # by configuration. Tests also use temporary directories under .runtime.
        if _RUNTIME.resolve() not in self.directory.resolve().parents:
            raise ValueError('照片目录必须位于本项目 .runtime 内。')
        self.max_photos = min(100, max(1, int(config.get('max_photos', 100))))
        self.max_bytes = min(5 * 1024 * 1024, max(1024, int(config.get('max_photo_bytes', 5 * 1024 * 1024))))
        self._lock = threading.Lock()

    def _prepare(self):
        for item in [self.directory] + list(self.directory.parents):
            if item.is_symlink():
                raise ValueError('照片目录不能包含符号链接。')
            if item == _RUNTIME.parent:
                break
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(str(self.directory), 0o700)

    def _path(self, photo_id, suffix):
        if not isinstance(photo_id, str) or not _ID.fullmatch(photo_id):
            raise ValueError('照片标识无效。')
        return self.directory / (photo_id + suffix)

    @staticmethod
    def _atomic(path, payload):
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def save(self, observation, composition, detector):
        jpeg = observation.get('data')
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= self.max_bytes or
                not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9')):
            raise ValueError('原始照片不是有效范围内的 JPEG。')
        photo_id = str(uuid.uuid4())
        metadata = {'id': photo_id, 'url': '/api/photos/' + photo_id + '.jpg',
                    'source': 'onboard_rgb', 'image_sha256': hashlib.sha256(jpeg).hexdigest(),
                    'stamp': observation['stamp'], 'ros_stamp': observation['ros_stamp'],
                    'width': observation['width'], 'height': observation['height'],
                    'composition': copy.deepcopy(composition), 'detector': copy.deepcopy(detector),
                    'phone_saved': False, 'storage': 'host_persistent',
                    'pose': {'position': copy.deepcopy(observation['position']), 'yaw': observation['yaw']}}
        raw = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')
        if len(raw) > 65536:
            raise ValueError('照片元数据过大。')
        with self._lock:
            self._prepare()
            if len(list(self.directory.glob('*.jpg'))) >= self.max_photos:
                raise ValueError('照片存储已满，请先整理本地照片；未覆盖已有照片。')
            jpg_path, json_path = self._path(photo_id, '.jpg'), self._path(photo_id, '.json')
            try:
                self._atomic(jpg_path, jpeg)
                self._atomic(json_path, raw)
                directory_fd = os.open(str(self.directory), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except Exception:
                for path in (jpg_path, json_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                raise
        return metadata

    @staticmethod
    def _read(path, limit):
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError('照片文件无效或过大。')
            result = handle.read(limit + 1)
            if len(result) > limit:
                raise ValueError('照片文件过大。')
            return result

    def get(self, photo_id):
        with self._lock:
            # _prepare checks path components on reads too, including restarts.
            self._prepare()
            jpeg = self._read(self._path(photo_id, '.jpg'), self.max_bytes)
            metadata = json.loads(self._read(self._path(photo_id, '.json'), 65536).decode('utf-8'))
            if (metadata.get('id') != photo_id or metadata.get('source') != 'onboard_rgb' or
                    metadata.get('url') != '/api/photos/' + photo_id + '.jpg' or
                    metadata.get('image_sha256') != hashlib.sha256(jpeg).hexdigest()):
                raise ValueError('照片完整性校验失败。')
            return jpeg, metadata

    def remove(self, photo_id):
        with self._lock:
            for suffix in ('.jpg', '.json'):
                try:
                    self._path(photo_id, suffix).unlink()
                except FileNotFoundError:
                    pass
