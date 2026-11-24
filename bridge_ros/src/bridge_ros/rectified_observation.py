"""Rectify the exact frozen RGB bytes and project into that same pixel plane."""
import numpy as np

from .visual_geometry import build_projection


def rectified_observation(frame, points, camera_from_world, intrinsic, distortion, model, map_cache):
    # This also validates the model, its injective branch and measured returns.
    raw = build_projection(points, camera_from_world, intrinsic, frame['width'], frame['height'],
                           distortion=distortion, distortion_model=model)
    if model == 'elastic_pinhole':
        # The caller has explicitly selected the original raw-image pinhole
        # approximation. Keep empty D and the original bytes; never report
        # unknown physical distortion as a measured zero or rectified pixels.
        return frame['data'], raw, (), False
    if not any(abs(value) > 1e-10 for value in distortion):
        return frame['data'], raw, tuple(distortion), False
    import cv2
    decoded = cv2.imdecode(np.frombuffer(frame['data'], dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (frame['height'], frame['width']):
        raise RuntimeError('冻结 RGB 解码尺寸与相机标定不一致，不能校正另一帧图像')
    # Keep the same K, image size and optical frame: no crop, resize or invented
    # zero coefficients. The zero D below describes the actual rectified pixels.
    key = (frame['width'], frame['height'], intrinsic.tobytes(), tuple(distortion), model)
    maps = map_cache.get(key)
    if maps is None:
        maps = cv2.initUndistortRectifyMap(intrinsic, np.asarray(distortion, dtype=float),
                                         np.eye(3), intrinsic,
                                         (frame['width'], frame['height']), cv2.CV_32FC1)
        map_cache.clear()
        map_cache[key] = maps
    pixels = cv2.remap(decoded, maps[0], maps[1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('校正后的冻结 RGB 编码失败')
    # Only raw-visible, valid-branch returns can support a rectified pixel.
    # Still preserve the entire measured cloud for clustering and clearance.
    projected = build_projection(raw['points_world'], camera_from_world, intrinsic,
                                 frame['width'], frame['height'])
    projected['points_world_all'] = raw['points_world_all']
    return encoded.tobytes(), projected, (0.,) * 5, True
