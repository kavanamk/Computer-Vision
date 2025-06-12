import cv2
import numpy as np
import json

WINDOW_TITLE = "2D Transform Editor"
RECTANGLE_DIMENSIONS = (120, 80)

class EditableRectangle:
    def __init__(self, position, transformation_type='translation'):
        self.position = np.array(position, dtype=np.float32)
        self.width, self.height = RECTANGLE_DIMENSIONS
        self.transformation_type = transformation_type
        self.transformation_matrix = np.eye(3, dtype=np.float32)
        self.translate(position[0], position[1])

    def get_corners(self):
        half_width, half_height = self.width / 2, self.height / 2
        corners = np.array([
            [-half_width, -half_height, 1],
            [ half_width, -half_height, 1],
            [ half_width,  half_height, 1],
            [-half_width,  half_height, 1],
        ]).T
        transformed_corners = self.transformation_matrix @ corners
        return (transformed_corners[:2] / transformed_corners[2]).T.astype(np.float32)

    def translate(self, dx, dy):
        translation_matrix = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.transformation_matrix = translation_matrix @ self.transformation_matrix

    def apply_transformation(self, matrix):
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0, 0, 1]])
        self.transformation_matrix = matrix @ self.transformation_matrix

    def to_dict(self):
        return {
            'transformation_matrix': self.transformation_matrix.tolist(),
            'width': self.width,
            'height': self.height
        }

    @staticmethod
    def from_dict(data):
        rectangle = EditableRectangle((0, 0))
        rectangle.transformation_matrix = np.array(data['transformation_matrix'], dtype=np.float32)
        rectangle.width = data['width']
        rectangle.height = data['height']
        return rectangle

def convert_to_homogeneous(matrix):
    return np.vstack([matrix, [0, 0, 1]])

# Global Variables
rectangles_list = []
dragging_active = False
active_rectangle = None
active_corner = None
previous_point = None
current_mode = 'translation'
MODES = ['translation', 'rigid', 'similarity', 'affine', 'perspective']

# Rubber-band for rectangle creation
rubber_band_active = False
start_point = None
end_point = None

def find_nearest_corner(rectangle, point):
    corners = rectangle.get_corners()
    for idx, corner in enumerate(corners):
        if np.linalg.norm(corner - point) < 20:
            return idx
    return None

def mouse_event_handler(event, x, y, flags, param):
    global dragging_active, active_rectangle, active_corner, previous_point
    global rubber_band_active, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            rubber_band_active = True
            start_point = (x, y)
            end_point = (x, y)
        else:
            for rectangle in reversed(rectangles_list):
                corner_idx = find_nearest_corner(rectangle, np.array([x, y], dtype=np.float32))
                if corner_idx is not None:
                    active_rectangle = rectangle
                    active_corner = corner_idx
                    dragging_active = True
                    previous_point = np.array([x, y], dtype=np.float32)
                    break

    elif event == cv2.EVENT_MOUSEMOVE:
        if rubber_band_active:
            end_point = (x, y)
        elif dragging_active and active_rectangle is not None:
            current_point = np.array([x, y], dtype=np.float32)
            old_corners = active_rectangle.get_corners()
            new_corners = old_corners.copy()
            new_corners[active_corner] = current_point

            if current_mode == 'translation':
                dx, dy = current_point - previous_point
                active_rectangle.translate(dx, dy)

            elif current_mode == 'rigid':
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
                matrix = np.eye(3, dtype=np.float32)
                matrix[:2, :2] = R
                matrix[:2, 2] = dst_center - R @ src_center
                active_rectangle.apply_transformation(matrix)

            elif current_mode == 'similarity':
                src = old_corners[:3].astype(np.float32)
                dst = new_corners[:3].astype(np.float32)
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
            
                src_centered = src - src_center
                dst_centered = dst - dst_center
            
                matrix, _ = cv2.estimateAffinePartial2D(src_centered, dst_centered, method=cv2.LMEDS)
                if matrix is not None:
                    T1 = np.array([[1, 0, -src_center[0]],
                                   [0, 1, -src_center[1]],
                                   [0, 0, 1]], dtype=np.float32)
                    T2 = np.array([[1, 0, dst_center[0]],
                                   [0, 1, dst_center[1]],
                                   [0, 0, 1]], dtype=np.float32)
                    H = T2 @ convert_to_homogeneous(matrix) @ T1
                    active_rectangle.apply_transformation(H)
            
            elif current_mode == 'affine':
                indices = [(active_corner + i) % 4 for i in range(3)]
                src = old_corners[indices].astype(np.float32)
                dst = new_corners[indices].astype(np.float32)
            
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
            
                src_centered = src - src_center
                dst_centered = dst - dst_center
            
                matrix = cv2.getAffineTransform(src_centered, dst_centered)
                T1 = np.array([[1, 0, -src_center[0]],
                               [0, 1, -src_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                T2 = np.array([[1, 0, dst_center[0]],
                               [0, 1, dst_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                H = T2 @ convert_to_homogeneous(matrix) @ T1
                active_rectangle.apply_transformation(H)

            elif current_mode == 'perspective':
                src = old_corners.astype(np.float32)
                dst = new_corners.astype(np.float32)
            
                src_center = np.mean(src, axis=0)
                dst_center = np.mean(dst, axis=0)
            
                src_centered = src - src_center
                dst_centered = dst - dst_center
            
                matrix = cv2.getPerspectiveTransform(src_centered, dst_centered)
                T1 = np.array([[1, 0, -src_center[0]],
                               [0, 1, -src_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                T2 = np.array([[1, 0, dst_center[0]],
                               [0, 1, dst_center[1]],
                               [0, 0, 1]], dtype=np.float32)
                H = T2 @ matrix @ T1
                active_rectangle.transformation_matrix = H @ active_rectangle.transformation_matrix

            previous_point = current_point

    elif event == cv2.EVENT_LBUTTONUP:
        if rubber_band_active:
            rubber_band_active = False
            x1, y1 = start_point
            x2, y2 = end_point
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if width > 5 and height > 5:
                rectangle = EditableRectangle(center, transformation_type=current_mode)
                rectangle.width = width
                rectangle.height = height
                rectangles_list.append(rectangle)
            start_point = None
            end_point = None
        dragging_active = False
        active_rectangle = None
        active_corner = None

def render():
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
    for rectangle in rectangles_list:
        corners = rectangle.get_corners().astype(np.int32)
        cv2.polylines(canvas, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
        for pt in corners:
            cv2.circle(canvas, pt, 5, (255, 0, 0), -1)
    if rubber_band_active and start_point and end_point:
        cv2.rectangle(canvas, start_point, end_point, (0, 255, 0), 1)
    cv2.putText(canvas, f"Mode: {current_mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return canvas

def save_rectangles_data(filename="rectangles.json"):
    data = [rect.to_dict() for rect in rectangles_list]
    with open(filename, 'w') as file:
        json.dump(data, file)
    print("Saved to", filename)

def load_rectangles_data(filename="rectangles.json"):
    global rectangles_list
    with open(filename, 'r') as file:
        data = json.load(file)
        rectangles_list = [EditableRectangle.from_dict(d) for d in data]
    print("Loaded from", filename)

def start():
    global current_mode
    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, mouse_event_handler)

    print("Controls:")
    print("  Shift + Drag: Create rectangle")
    print("  Drag corner: Transform rectangle")
    print("  1-5: Switch mode (1=Translation, 2=Rigid, 3=Similarity, 4=Affine, 5=Perspective)")
    print("  s: Save, l: Load, q: Quit")

    while True:
        canvas = render()
        cv2.imshow(WINDOW_TITLE, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_rectangles_data()
        elif key == ord('l'):
            load_rectangles_data()
        elif key in [ord(str(i)) for i in range(1, 6)]:
            current_mode = MODES[int(chr(key)) - 1]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    start()
