import cv2
import numpy as np
import json

WINDOW_NAME = "2D Transform Editor"
RECT_SIZE = (120, 80)

class TransformableRectangle:
    def __init__(self, center, transform_type='translation'):
        self.center = np.array(center, dtype=np.float32)
        self.width, self.height = RECT_SIZE
        self.transform_type = transform_type
        self.matrix = np.eye(3, dtype=np.float32)
        self.move(center[0], center[1])

    def get_corners(self):
        hw, hh = self.width / 2, self.height / 2
        corners = np.array([
            [-hw, -hh, 1],
            [ hw, -hh, 1],
            [ hw,  hh, 1],
            [-hw,  hh, 1],
        ]).T
        transformed = self.matrix @ corners
        return (transformed[:2] / transformed[2]).T.astype(np.float32)

    def move(self, dx, dy):
        t = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = t @ self.matrix

    def apply_transformation(self, M):
        if M.shape == (2, 3):
            M = np.vstack([M, [0, 0, 1]])
        self.matrix = M @ self.matrix

    def to_dict(self):
        return {
            'matrix': self.matrix.tolist(),
            'width': self.width,
            'height': self.height
        }

    @staticmethod
    def from_dict(d):
        rect = TransformableRectangle((0, 0))
        rect.matrix = np.array(d['matrix'], dtype=np.float32)
        rect.width = d['width']
        rect.height = d['height']
        return rect

def to_homogeneous(M):
    return np.vstack([M, [0, 0, 1]])

# Globals
rectangles = []
dragging = False
selected_rect = None
selected_corner = None
prev_point = None
mode = 'translation'
MODE_MAP = ['translation', 'rigid', 'similarity', 'affine', 'perspective']

# Rubber-band creation
rubber_band_mode = False
start_point = None
end_point = None

def get_nearest_corner(rect, point):
    corners = rect.get_corners()
    for i, corner in enumerate(corners):
        if np.linalg.norm(corner - point) < 20:
            return i
    return None

def mouse_callback(event, x, y, flags, param):
    global dragging, selected_rect, selected_corner, prev_point
    global rubber_band_mode, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            rubber_band_mode = True
            start_point = (x, y)
            end_point = (x, y)
        else:
            for rect in reversed(rectangles):
                corner_idx = get_nearest_corner(rect, np.array([x, y], dtype=np.float32))
                if corner_idx is not None:
                    selected_rect = rect
                    selected_corner = corner_idx
                    dragging = True
                    prev_point = np.array([x, y], dtype=np.float32)
                    break

    elif event == cv2.EVENT_MOUSEMOVE:
        if rubber_band_mode:
            end_point = (x, y)
        elif dragging and selected_rect is not None:
            curr_point = np.array([x, y], dtype=np.float32)
            old_corners = selected_rect.get_corners()
            new_corners = old_corners.copy()
            new_corners[selected_corner] = curr_point

            if mode == 'translation':
                dx, dy = curr_point - prev_point
                selected_rect.move(dx, dy)

            elif mode == 'rigid':
                src = old_corners[:2].astype(np.float32)
                dst = new_corners[:2].astype(np.float32)
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
                src_centered = src - src_center
                dst_centered = dst - dst_center
                H = src_centered.T @ dst_centered
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                M = np.eye(3, dtype=np.float32)
                M[:2, :2] = R
                M[:2, 2] = dst_center - R @ src_center
                selected_rect.apply_transformation(M)

            elif mode == 'similarity':
                src = old_corners[:3].astype(np.float32)
                dst = new_corners[:3].astype(np.float32)
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
            
                src_centered = src - src_center
                dst_centered = dst - dst_center
            
                M, _ = cv2.estimateAffinePartial2D(src_centered, dst_centered, method=cv2.LMEDS)
                if M is not None:
                    T1 = np.array([[1, 0, -src_center[0]],
                                   [0, 1, -src_center[1]],
                                   [0, 0, 1]], dtype=np.float32)
                    T2 = np.array([[1, 0, dst_center[0]],
                                   [0, 1, dst_center[1]],
                                   [0, 0, 1]], dtype=np.float32)
                    H = T2 @ to_homogeneous(M) @ T1
                    selected_rect.apply_transformation(H)
            
            elif mode == 'affine':
                idxs = [(selected_corner + i) % 4 for i in range(3)]
                src = old_corners[idxs].astype(np.float32)
                dst = new_corners[idxs].astype(np.float32)
            
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
            
                src_centered = src - src_center
                dst_centered = dst - dst_center
            
                M = cv2.getAffineTransform(src_centered, dst_centered)
                T1 = np.array([[1, 0, -src_center[0]],
                               [0, 1, -src_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                T2 = np.array([[1, 0, dst_center[0]],
                               [0, 1, dst_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                H = T2 @ to_homogeneous(M) @ T1
                selected_rect.apply_transformation(H)

            elif mode == 'perspective':
                src = old_corners.astype(np.float32)
                dst = new_corners.astype(np.float32)
            
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
            
                src_centered = src - src_center
                dst_centered = dst - dst_center
            
                M = cv2.getPerspectiveTransform(src_centered, dst_centered)
                T1 = np.array([[1, 0, -src_center[0]],
                               [0, 1, -src_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                T2 = np.array([[1, 0, dst_center[0]],
                               [0, 1, dst_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                H = T2 @ M @ T1
                selected_rect.matrix = H @ selected_rect.matrix


            prev_point = curr_point

    elif event == cv2.EVENT_LBUTTONUP:
        if rubber_band_mode:
            rubber_band_mode = False
            x1, y1 = start_point
            x2, y2 = end_point
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if width > 5 and height > 5:
                rect = TransformableRectangle(center, transform_type=mode)
                rect.width = width
                rect.height = height
                rectangles.append(rect)
            start_point = None
            end_point = None
        dragging = False
        selected_rect = None
        selected_corner = None

def draw():
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
    for rect in rectangles:
        corners = rect.get_corners().astype(np.int32)
        cv2.polylines(canvas, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
        for pt in corners:
            cv2.circle(canvas, pt, 5, (255, 0, 0), -1)
    if rubber_band_mode and start_point and end_point:
        cv2.rectangle(canvas, start_point, end_point, (0, 255, 0), 1)
    cv2.putText(canvas, f"Mode: {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return canvas

def save_rectangles(filename="rectangles.json"):
    data = [r.to_dict() for r in rectangles]
    with open(filename, 'w') as f:
        json.dump(data, f)
    print("Saved to", filename)

def load_rectangles(filename="rectangles.json"):
    global rectangles
    with open(filename, 'r') as f:
        data = json.load(f)
        rectangles = [TransformableRectangle.from_dict(d) for d in data]
    print("Loaded from", filename)

def main():
    global mode
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Controls:")
    print("  Shift + Drag: Create rectangle")
    print("  Drag corner: Transform rectangle")
    print("  1-5: Switch mode (1=Translation, 2=Rigid, 3=Similarity, 4=Affine, 5=Perspective)")
    print("  s: Save, l: Load, q: Quit")

    while True:
        canvas = draw()
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_rectangles()
        elif key == ord('l'):
            load_rectangles()
        elif key in [ord(str(i)) for i in range(1, 6)]:
            mode = MODE_MAP[int(chr(key)) - 1]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
