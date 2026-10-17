"""Read-only access to the supplied SQLite ROS bags. No raw data is rewritten."""
from pathlib import Path
import sqlite3
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from .geometry import transform_matrix

CAMERA = '/blackfly_s/cam0/image_rectified'
LIDAR = '/velodyne/points_raw'
RADAR = '/PointCloudDetection'


class Bag:
    def __init__(self, path):
        path = Path(path).resolve()
        if path.is_dir():
            files = sorted(path.glob('*.db3'))
            if len(files) != 1:
                raise ValueError('Expected one .db3 per supplied bag directory')
            path = files[0]
        self.path = path
        self.db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, check_same_thread=False)
        self.topics = {name: (tid, get_message(typ), typ) for tid, name, typ in
                       self.db.execute('SELECT id,name,type FROM topics')}
        self.index = {}
        for name, (tid, _, _) in self.topics.items():
            self.index[name] = np.array(self.db.execute(
                'SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id',
                (tid,)).fetchall(), dtype=np.int64).reshape(-1, 2)
        self.start, self.end = self.db.execute('SELECT MIN(timestamp),MAX(timestamp) FROM messages').fetchone()

    def close(self):
        self.db.close()

    def read(self, name, index):
        row_id, timestamp = self.index[name][index]
        raw = self.db.execute('SELECT data FROM messages WHERE id=?', (int(row_id),)).fetchone()[0]
        return int(timestamp), deserialize_message(raw, self.topics[name][1])

    def nearest(self, name, timestamp, tolerance_ns=100_000_000):
        entries = self.index.get(name)
        if entries is None or not len(entries):
            return None
        i = int(np.searchsorted(entries[:, 1], timestamp))
        choices = [j for j in (i - 1, i) if 0 <= j < len(entries)]
        best = min(choices, key=lambda j: abs(int(entries[j, 1]) - timestamp))
        if abs(int(entries[best, 1]) - timestamp) > tolerance_ns:
            return None
        return self.read(name, best)

    def static_transforms(self):
        transforms = {}
        for i in range(len(self.index.get('/tf_static', []))):
            for t in self.read('/tf_static', i)[1].transforms:
                transforms[t.child_frame_id] = t
        return list(transforms.values())

    def lidar_from(self, child):
        for t in self.static_transforms():
            if t.header.frame_id == 'velodyne' and t.child_frame_id == child:
                p, q = t.transform.translation, t.transform.rotation
                return transform_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        raise ValueError(f'Missing velodyne -> {child} calibration in bag')

    def messages(self):
        by_id = {v[0]: (name, v[1]) for name, v in self.topics.items()}
        for tid, timestamp, raw in self.db.execute(
                'SELECT topic_id,timestamp,data FROM messages ORDER BY timestamp,id'):
            name, typ = by_id[tid]
            yield name, timestamp, deserialize_message(raw, typ)
