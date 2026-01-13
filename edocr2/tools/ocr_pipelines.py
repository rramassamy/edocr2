"""
edocr2 OCR Pipelines - Version nettoyée et optimisée
Fork: github.com/rramassamy/edocr2
Modifications:
- Suppression de tout le code debug (cv2.imshow, prints, etc.)
- Fix du bug numpy array dans symbol_search (if box: → if box is not None:)
- Fix division par zéro (std, h_, norm, max_dim, rect dimensions)
- Nettoyage des imports et du code mort
- Ajout de try/except pour robustesse
"""

import cv2
import math
import os
import numpy as np


def read_alphabet(keras_path):
    """Lit l'alphabet depuis le fichier .txt associé au modèle"""
    txt_path = os.path.splitext(keras_path)[0] + '.txt'
    with open(txt_path, 'r') as file:
        content = file.readline().strip()
    return content


# ==================== Tables and Others Pipeline ====================

def ocr_img_cv2(image_cv2, language=None, psm=11):
    """Recognize text in an OpenCV image using pytesseract."""
    import pytesseract
    
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    if language:
        custom_config = f'--psm {psm} -l {language}'
    else:
        custom_config = f'--psm {psm}'

    ocr_data = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=pytesseract.Output.DICT)

    result = []
    all_text = ''
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():
            text_info = {
                'text': ocr_data['text'][i],
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            }
            all_text += ocr_data['text'][i]
            result.append(text_info)
    
    return result, all_text


def ocr_tables(tables, process_img, language=None):
    """OCR sur les tables/cartouches détectés"""
    results = []
    updated_tables = []

    tables = sorted(tables, key=lambda cluster_dict: next(iter(cluster_dict)).y * 10000 + next(iter(cluster_dict)).x, reverse=True)

    for table in tables:
        for b in table:
            img = process_img[b.y : b.y + b.h, b.x : b.x + b.w][:]
            result, all_text = ocr_img_cv2(img, language)
            if result == [] or len(all_text) < 5:
                continue
            else:
                for r in result:
                    r['left'] += b.x
                    r['top'] += b.y
                results.append(result)
                updated_tables.append(table)
    
    for table in updated_tables:
        for b in table:
            process_img[b.y : b.y + b.h, b.x : b.x + b.w][:] = 255
    
    return results, updated_tables, process_img


# ==================== GDT Pipeline ====================

def img_not_empty(roi, color_thres=100):
    """Vérifie si une ROI contient des pixels significatifs"""
    if roi is None or roi.size == 0:
        return False
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_roi)
    
    if (max_val - min_val) < color_thres:
        return False
    return True


def is_not_empty(img, boxes, color_thres):
    """Vérifie si les boxes contiennent du contenu"""
    for box in boxes:
        y1 = max(0, box.y + 2)
        y2 = max(y1 + 1, box.y + box.h - 4)
        x1 = max(0, box.x + 2)
        x2 = max(x1 + 1, box.x + box.w - 4)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0 or img_not_empty(roi, color_thres) == False:
            return False
    return True


def sort_gdt_boxes(boxes, y_thres=3):
    """Trie les boxes en ordre de lecture (gauche→droite, haut→bas)"""
    if not boxes:
        return boxes
    
    boxes.sort(key=lambda b: b.y)

    sorted_boxes = []
    current_line = []
    current_y = boxes[0].y

    for box in boxes:
        if abs(box.y - current_y) <= y_thres:
            current_line.append(box)
        else:
            current_line.sort(key=lambda b: b.x)
            sorted_boxes.extend(current_line)
            current_line = [box]
            current_y = box.y
    
    current_line.sort(key=lambda b: b.x)
    sorted_boxes.extend(current_line)
    
    return sorted_boxes


def recognize_gdt(img, block, recognizer):
    """Reconnaît les symboles GD&T dans un bloc"""
    if not block:
        return None
    
    y1 = max(0, block[0].y + 2)
    y2 = max(y1 + 1, block[0].y + block[0].h - 4)
    x1 = max(0, block[0].x + 2)
    x2 = max(x1 + 1, block[0].x + block[0].w - 4)
    roi = img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
    
    pred = recognizer.recognize(image=roi)

    for i in range(1, len(block)):
        new_line = block[i].y - block[i - 1].y > 5
        y1 = max(0, block[i].y)
        y2 = max(y1 + 1, block[i].y + block[i].h)
        x1 = max(0, block[i].x)
        x2 = max(x1 + 1, block[i].x + block[i].w)
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
        
        p = recognizer.recognize(image=roi)
        if new_line:
            pred += '\n' + p
        else:
            pred += '|' + p
    
    if any(char.isdigit() for char in pred):
        return pred
    else:
        return None


def ocr_gdt(img, gdt_boxes, recognizer):
    """Pipeline OCR pour les GD&T"""
    updated_gdts = []
    results = []
    
    if gdt_boxes:
        for block in gdt_boxes:
            for _, bl_list in block.items():
                if is_not_empty(img, bl_list, 50):
                    sorted_block = sort_gdt_boxes(bl_list, 3)
                    pred = recognize_gdt(img, sorted_block, recognizer)
                    if pred:
                        updated_gdts.append(block)
                        results.append([pred, (sorted_block[0].x, sorted_block[0].y)])
    
    for gdt in updated_gdts:
        for g in gdt.values():
            for b in g:
                y1 = max(0, b.y - 5)
                y2 = min(img.shape[0], b.y + b.h + 10)
                x1 = max(0, b.x - 5)
                x2 = min(img.shape[1], b.x + b.w + 10)
                img[y1:y2, x1:x2] = 255
    
    return results, updated_gdts, img


# ==================== Dimension Pipeline ====================

class Pipeline:
    """Pipeline de détection et reconnaissance de dimensions"""
    
    def __init__(self, detector, recognizer, alphabet_dimensions, cluster_t=20, scale=2, matching_t=0.6, max_size=1024, language='eng'):
        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer
        self.max_size = max_size
        self.language = language
        self.alphabet_dimensions = alphabet_dimensions
        self.cluster_t = cluster_t
        self.matching_t = matching_t

    def symbol_search(self, img, dimensions, folder_code='u2300', char='⌀'):
        """Recherche et associe les symboles (diamètre, etc.) aux dimensions"""
        
        def template_matching(img_, cnts, folder_path, thres, angle, xy2, rotate):
            angle = math.radians(angle)
            box_points = None
            
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                # FIX: Éviter division par zéro
                if h <= 0 or w <= 0:
                    continue
                if h > img_.shape[0] * 0.3:
                    img_2 = img_[y:y + h, x:x + w]
                    if img_2.size == 0:
                        continue
                    y_pad, x_pad = int(img_2.shape[0] * 0.3), 40
                    pad_img = cv2.copyMakeBorder(img_2, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    
                    for file in os.listdir(folder_path):
                        symb = cv2.imread(os.path.join(folder_path, file))
                        if symb is None:
                            continue
                        if rotate:
                            symb = cv2.rotate(symb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        gray = cv2.cvtColor(symb, cv2.COLOR_BGR2GRAY)
                        _, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                        contours_smb, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if not contours_smb:
                            continue
                        
                        x_, y_, w_, h_ = cv2.boundingRect(contours_smb[0])
                        
                        # FIX: Éviter division par zéro
                        if h_ <= 0 or w_ <= 0:
                            continue
                        
                        symb_img = symb[y_:y_ + h_, x_:x_ + w_]
                        if symb_img.size == 0:
                            continue
                        
                        scale_factor = h / h_
                        
                        if scale_factor < 2 and scale_factor > 0:
                            scaled_symb = cv2.resize(symb_img, (0, 0), fx=scale_factor, fy=scale_factor)
                            
                            # Vérifier que le template n'est pas plus grand que l'image
                            if scaled_symb.shape[0] > pad_img.shape[0] or scaled_symb.shape[1] > pad_img.shape[1]:
                                continue
                            if scaled_symb.size == 0:
                                continue
                            
                            result = cv2.matchTemplate(pad_img, scaled_symb, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(result)
                            
                            if max_val >= thres:
                                local = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                                box_points = [
                                    (xy2[0] + math.cos(angle)*local[0][0] - math.sin(angle)*local[0][1],
                                     xy2[1] + math.cos(angle)*local[0][1] + math.sin(angle)*local[0][0]),
                                    (xy2[0] + math.cos(angle)*local[1][0] - math.sin(angle)*local[1][1],
                                     xy2[1] + math.cos(angle)*local[1][1] + math.sin(angle)*local[1][0]),
                                    (xy2[0] + math.cos(angle)*local[2][0] - math.sin(angle)*local[2][1],
                                     xy2[1] + math.cos(angle)*local[2][1] + math.sin(angle)*local[2][0]),
                                    (xy2[0] + math.cos(angle)*local[3][0] - math.sin(angle)*local[3][1],
                                     xy2[1] + math.cos(angle)*local[3][1] + math.sin(angle)*local[3][0])
                                ]
                                thres = max_val
            return box_points

        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        
        old_dim, boxes = [], []
        folder_path = os.path.join('edocr2/tools/symbol_match', folder_code)
        
        # Vérifier que le dossier existe
        if not os.path.exists(folder_path):
            return dimensions
        
        for dim in dimensions:
            if char in dim[0]:
                continue
            
            try:
                rect = cv2.minAreaRect(np.array(dim[1], dtype=np.float32))
            except Exception:
                continue
            
            # FIX: Éviter dimensions nulles
            if min(rect[1]) <= 0 or max(rect[1]) <= 0:
                continue
            
            if len(dim[0]) == 1:
                w_multiplier = 1.3
                min_rect = min(rect[1])
                h_multiplier = max([2 * min_rect, 300]) / min_rect if min_rect > 0 else 1
                img_, cnts, angle = postprocess_detection(img, dim[1], w_multiplier, h_multiplier, 5)
                rotate = True
            else:
                max_rect = max(rect[1])
                w_multiplier = max([2 * max_rect, 300]) / max_rect if max_rect > 0 else 1
                h_multiplier = 1.3
                img_, cnts, angle = postprocess_detection(img, dim[1], w_multiplier, h_multiplier, 5)
                rotate = False
            
            if img_ is None or img_.size == 0:
                continue
            
            scaled_rect = (rect[0], (img_.shape[0], img_.shape[1]), angle - 90)
            xy2 = (
                rect[0][0] - scaled_rect[1][1]/2*math.cos(math.radians(angle)) + scaled_rect[1][0]/2*math.sin(math.radians(angle)),
                rect[0][1] - scaled_rect[1][1]/2*math.sin(math.radians(angle)) - scaled_rect[1][0]/2*math.cos(math.radians(angle))
            )
            
            box = template_matching(img_, cnts, folder_path, self.matching_t, angle, xy2, rotate)
            
            # FIX: Utiliser "is not None" au lieu de vérification booléenne directe sur numpy array
            if box is not None:
                try:
                    pts = np.array([(box[0]), (box[1]), (box[2]), (box[3])]).astype(np.int64)
                    poly2 = Polygon(box)
                    poly1 = Polygon(cv2.boxPoints(rect))
                    merged_poly = unary_union([poly1, poly2])
                    final_box = merged_poly.minimum_rotated_rectangle.exterior.coords[0:4]
                    boxes.append(final_box)
                    old_dim.append(dim)
                except Exception:
                    continue
        
        for o in old_dim:
            dimensions.remove(o)
        
        if boxes:
            try:
                boxes = group_polygons_by_proximity(boxes, eps=self.cluster_t)
                new_group = [box for box in boxes]
                new_dimensions, _, _ = self.recognize_dimensions(np.int32(new_group), np.array(img))
                
                for nd in new_dimensions:
                    if char in nd[0]:
                        dimensions.append(nd)
                    elif nd[0][0] in set('0,).D:Z°Bx'):
                        dimensions.append((char + nd[0][1:], nd[1]))
                    else:
                        dimensions.append((char + nd[0], nd[1]))
            except Exception:
                pass
        
        return dimensions

    def detect(self, img, detection_kwargs=None):
        """Détecte les zones de texte dans l'image"""
        from edocr2.keras_ocr.tools import adjust_boxes

        # FIX: Éviter division par zéro si image vide
        if img is None or img.size == 0:
            return [[]]
        
        max_dim = np.max((img.shape[0], img.shape[1]))
        if max_dim == 0:
            return [[]]
        
        if max_dim < self.max_size / self.scale:
            scale = self.scale
        else:
            scale = self.max_size / max_dim

        if detection_kwargs is None:
            detection_kwargs = {}
        
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        if new_size[0] <= 0 or new_size[1] <= 0:
            return [[]]
        
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        box_groups = self.detector.detect(images=[img], **detection_kwargs)
        box_groups = [
            adjust_boxes(boxes=boxes, boxes_format="boxes", scale=1 / scale)
            if scale != 1
            else boxes
            for boxes, scale in zip(box_groups, [scale])
        ]
        return box_groups

    def ocr_the_rest(self, img, lang):
        """OCR Tesseract pour le texte restant"""
        
        def sort_boxes_by_centers(boxes, y_threshold=20):
            if not boxes:
                return ""
            sorted_boxes = sorted(boxes, key=lambda box: (box['top'], box['left']))
            final_sorted_text = ""
            current_line = []
            current_y = sorted_boxes[0]['top']

            for box in sorted_boxes:
                if abs(box['top'] - current_y) <= y_threshold:
                    current_line.append(box)
                else:
                    current_line = sorted(current_line, key=lambda b: b['left'])
                    line_text = ' '.join([b['text'] for b in current_line])
                    final_sorted_text += line_text + '\n'
                    current_line = [box]
                    current_y = box['top']

            current_line = sorted(current_line, key=lambda b: b['left'])
            line_text = ' '.join([b['text'] for b in current_line])
            final_sorted_text += line_text

            return final_sorted_text
    
        if img is None or img.size == 0:
            return ''
        
        results, _ = ocr_img_cv2(img, lang)
        if results:
            text = sort_boxes_by_centers(results)
            return text
        return ''

    def dimension_criteria(self, img):
        """Vérifie si le contenu ressemble à une dimension"""
        if img is None or img.size == 0:
            return False
        
        pred_nor = self.ocr_the_rest(img, 'nor')
        pred_eng = self.ocr_the_rest(img, 'eng')
        allowed_exceptions_nor = set('-.»Ø,/!«Æ()Å:\'"[];|"?Ö=*Ä"&É<>+$£%—€øåæöéIZNOoPXiLlk \n')
        allowed_exceptions_eng = set('?—!@#~;¢«#_%\\&€$»[é]®§¥©\'™="~\'£<*""I|ZNOXiLlk \n')
        ok_nor = all(char in set(self.alphabet_dimensions) or char in allowed_exceptions_nor for char in pred_nor)
        ok_eng = all(char in set(self.alphabet_dimensions) or char in allowed_exceptions_eng for char in pred_eng)
        if ok_nor or ok_eng or len(pred_eng) < 2 or len(pred_nor) < 2:
            return True
        return False

    def recognize_dimensions(self, box_groups, img):
        """Reconnaît les dimensions dans les boxes détectées"""
        predictions = []
        predictions_pyt = []
        other_info = []

        def adjust_padding(img):
            if img is None or img.size == 0:
                return img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if cnts:
                x, y, w, h = cv2.boundingRect(np.concatenate(cnts))
                if w > 0 and h > 0:
                    img = img[y:y+h, x:x+w]
                    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            return img
        
        def adjust_stroke(img):
            if img is None or img.size == 0:
                return img
            
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_, 200, 255, cv2.THRESH_BINARY_INV)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            final_img = np.full_like(img_, 255)
            
            stroke_averages = []
            subimages = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0:
                    continue
                subimage = np.full_like(img_, 255)
                subimage[y:y+h, x:x+w] = img_[y:y+h, x:x+w]
                subimages.append(subimage)
                counts = []

                for i in range(y, y + h):
                    row = subimage[i, :]
                    classified = row < 180
                    current_length = 0

                    for val in classified:
                        if val:
                            current_length += 1
                        else:
                            if current_length > 0:
                                counts.extend([current_length])
                                current_length = 0

                    if current_length > 0:
                        counts.extend([current_length])

                outliers = find_outliers(counts, 1.5)
                filtered_counts = [c for c in counts if c not in outliers]
                
                avg_stroke = np.mean(filtered_counts) if filtered_counts else 0
                stroke_averages.append(avg_stroke)

            if not stroke_averages:
                return img

            outliers = find_outliers(stroke_averages, 3)
            if len(outliers) > 0 or any(st < 2.5 for st in stroke_averages):
                for i in range(len(subimages)):
                    if i >= len(stroke_averages):
                        continue
                    processed_subimage = subimages[i]
                    
                    if len(outliers) > 0 and len(stroke_averages) < 2:
                        if stroke_averages[i] < np.min(outliers) or stroke_averages[i] < 2.5:
                            kernel = np.ones((3, 3), np.uint8)
                            processed_subimage = cv2.erode(processed_subimage, kernel, iterations=1)

                    elif len(stroke_averages) == 2:
                        if np.max(stroke_averages) - stroke_averages[i] > 1.5 or stroke_averages[i] < 2.5:
                            kernel = np.ones((3, 3), np.uint8)
                            processed_subimage = cv2.erode(processed_subimage, kernel, iterations=1)
                    else:
                        if stroke_averages[i] < 2.5:
                            kernel = np.ones((3, 3), np.uint8)
                            processed_subimage = cv2.erode(processed_subimage, kernel, iterations=1)

                    _, thresh = cv2.threshold(processed_subimage, 200, 255, cv2.THRESH_BINARY_INV)
                    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if cnts:
                        x, y, w, h = cv2.boundingRect(cnts[0])
                        if w > 0 and h > 0:
                            final_img[y:y+h, x:x+w] = processed_subimage[y:y+h, x:x+w]
                
                return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
            
            return img

        def pad_image(img, pad_percent):
            if img is None or img.size == 0:
                return img
            y_pad, x_pad = int(img.shape[0] * pad_percent), int(img.shape[1] * pad_percent)
            pad_img = cv2.copyMakeBorder(img, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            return pad_img
        
        for box in box_groups:
            try:
                img_croped, cnts, _ = postprocess_detection(img, box)
            except Exception:
                continue
            
            if img_croped is None or img_croped.size == 0:
                continue
            
            if len(cnts) == 1:
                img_croped = cv2.rotate(img_croped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                pred = self.recognizer.recognize(image=img_croped)
                if pred.isdigit():
                    predictions.append((pred, box))
            else:
                pytess_img = pad_image(img_croped, 0.3)
                if pytess_img is None or pytess_img.size == 0:
                    continue
                
                if self.dimension_criteria(pytess_img):
                    arr = check_tolerances(img_croped)
                    pred = ''
                    for img_ in arr:
                        img_ = adjust_padding(img_)
                        if img_ is None or img_.size == 0:
                            continue
                        if img_.shape[0] * img_.shape[1] > 1200:
                            img_ = adjust_stroke(img_)
                        pred_ = self.recognizer.recognize(image=img_) + ' '
                        if pred_ == ' ':
                            pred = self.recognizer.recognize(image=img_croped) + ' '
                            break
                        else:
                            pred += pred_
                        
                    if any(char.isdigit() for char in pred):
                        predictions.append((pred[:-1], box))
                    else:
                        pred_pyt = self.ocr_the_rest(pytess_img, self.language)
                        other_info.append((pred_pyt, box))
                else:
                    pred_pyt = self.ocr_the_rest(pytess_img, self.language)
                    other_info.append((pred_pyt, box))
        
        return predictions, other_info, predictions_pyt

    def ocr_img_patches(self, img, ol=0.05):
        """Découpe l'image en patches et effectue l'OCR"""
        if img is None or img.size == 0:
            return [], [], []
        
        # FIX: Éviter division par zéro
        if self.max_size <= 0:
            return [], [], []
        
        patches = (int(img.shape[1] / self.max_size + 2), int(img.shape[0] / self.max_size + 2))
        
        # FIX: Éviter division par zéro
        if patches[0] <= 0 or patches[1] <= 0:
            return [], [], []
        
        a_x = int((1 - ol) / (patches[0]) * img.shape[1])
        b_x = a_x + int(ol * img.shape[1])
        a_y = int((1 - ol) / (patches[1]) * img.shape[0])
        b_y = a_y + int(ol * img.shape[0])
        box_groups = []
        
        for i in range(0, patches[0]):
            for j in range(0, patches[1]):
                offset = (a_x * i, a_y * j)
                patch_boundary = (i * a_x + b_x, j * a_y + b_y)
                img_patch = img[offset[1]:patch_boundary[1], offset[0]:patch_boundary[0]]
                if img_patch.size > 0 and img_not_empty(img_patch, 100):
                    box_group = self.detect(img_patch)
                    for b in box_group:
                        for xy in b:
                            xy = xy + offset
                            box_groups.append(xy)
        
        if not box_groups:
            return [], [], []
        
        box_groups = group_polygons_by_proximity(box_groups, eps=self.cluster_t)
        box_groups = group_polygons_by_proximity(box_groups, eps=self.cluster_t - 5)
        
        new_group = [box for box in box_groups]
        dimensions, other_info, dimensions_pyt = self.recognize_dimensions(np.int32(new_group), np.array(img))
        dimensions = self.symbol_search(img, dimensions)
        
        return dimensions, other_info, dimensions_pyt


# ==================== Utility Functions ====================

def group_polygons_by_proximity(polygons, eps=20):
    """Groupe les polygones proches par proximité"""
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    if not polygons:
        return []

    def polygon_intersects_or_close(p1, p2, eps):
        try:
            poly1 = Polygon(p1)
            poly2 = Polygon(p2)
            if poly1.intersects(poly2):
                return True
            return poly1.distance(poly2) <= eps
        except Exception:
            return False

    n = len(polygons)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootX] = rootY
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygon_intersects_or_close(polygons[i], polygons[j], eps):
                union(i, j)
    
    grouped_polygons = {}
    for i in range(n):
        root = find(i)
        if root not in grouped_polygons:
            grouped_polygons[root] = []
        grouped_polygons[root].append(polygons[i])
    
    merged_polygons = []
    for group in grouped_polygons.values():
        try:
            merged_polygon = unary_union([Polygon(p) for p in group])
            
            if isinstance(merged_polygon, MultiPolygon):
                merged_polygon = unary_union(merged_polygon)
            if merged_polygon.is_empty:
                continue

            min_rotated_box = merged_polygon.minimum_rotated_rectangle.exterior.coords[0:4]
            merged_polygons.append(min_rotated_box)
        except Exception:
            continue
    
    return merged_polygons


def check_tolerances(img):
    """Détecte et sépare les tolérances empilées"""
    if img is None or img.size == 0:
        return [img] if img is not None else []
    
    img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flag = False
    tole = False
    top_line = 0
    bot_line = img_arr.shape[0] - 1
    
    # Find top and bottom line
    for i in range(0, img_arr.shape[0] - 1):
        for j in range(0, img_arr.shape[1] - 1):
            if img_arr[i, j] < 200:
                top_line = i
                flag = True
                break
        if flag:
            flag = False
            break
    
    for i in range(img_arr.shape[0] - 1, top_line, -1):
        for j in range(0, img_arr.shape[1] - 1):
            if img_arr[i, j] < 200:
                bot_line = i
                flag = True
                break
        if flag:
            break
    
    if top_line >= bot_line:
        return [img]
    
    stop_at = []
    for i in range(top_line, bot_line):
        for j in range(img_arr.shape[1] - 1, 0, -1):
            if img_arr[i, j] < 200:
                stop_at.append(img_arr.shape[1] - j)
                break
        else:
            stop_at.append(img_arr.shape[1])
    
    if not stop_at:
        return [img]
    
    tole_h_cut = top_line
    d = 0
    
    start_idx = int(0.3 * len(stop_at))
    end_idx = int(0.7 * len(stop_at))
    
    for idx, d in enumerate(stop_at[start_idx:end_idx]):
        if d > img_arr.shape[0] * 0.8:
            tole = True
            tole_h_cut = start_idx + idx + top_line + 1
            break
        else:
            tole = False

    if tole:
        if d < img_arr.shape[1]:
            tole_v_cut = None
            for j in range(img_arr.shape[1] - d, img_arr.shape[1]):
                if j < 0 or j >= img_arr.shape[1]:
                    continue
                start_row = int(0.3 * img_arr.shape[0])
                end_row = int(0.7 * img_arr.shape[0])
                if start_row < end_row and np.all(img_arr[start_row:end_row, j] > 200):
                    tole_v_cut = j + 2
                    break
            if tole_v_cut and tole_v_cut < img_arr.shape[1]:
                try:
                    measu_box = img_arr[:, :tole_v_cut]
                    up_tole_box = img_arr[:tole_h_cut, tole_v_cut:]
                    bot_tole_box = img_arr[tole_h_cut:, tole_v_cut:]
                    
                    result = []
                    if measu_box.size > 0:
                        result.append(cv2.cvtColor(measu_box, cv2.COLOR_GRAY2BGR))
                    if up_tole_box.size > 0:
                        result.append(cv2.cvtColor(up_tole_box, cv2.COLOR_GRAY2BGR))
                    if bot_tole_box.size > 0:
                        result.append(cv2.cvtColor(bot_tole_box, cv2.COLOR_GRAY2BGR))
                    return result if result else [img]
                except Exception:
                    return [img]
        else:
            try:
                up_text = img_arr[:tole_h_cut, :]
                bot_text = img_arr[tole_h_cut:, :]
                result = []
                if up_text.size > 0:
                    result.append(cv2.cvtColor(up_text, cv2.COLOR_GRAY2BGR))
                if bot_text.size > 0:
                    result.append(cv2.cvtColor(bot_text, cv2.COLOR_GRAY2BGR))
                return result if result else [img]
            except Exception:
                return [img]
    return [img]


def find_outliers(counts, t):
    """Trouve les outliers dans une liste de valeurs"""
    counts = np.array(counts)
    if len(counts) == 0:
        return np.array([])
    
    mean = np.mean(counts)
    std = np.std(counts)

    # FIX: Éviter division par zéro
    if std == 0:
        return np.array([])
    
    z_scores = (counts - mean) / std
    return counts[np.abs(z_scores) > t]


def postprocess_detection(img, box, w_multiplier=1.0, h_multiplier=1.0, angle_t=5):
    """Post-traitement des détections: rotation et crop"""
    
    def get_box_angle(box):
        exp_box = np.vstack((box[3], box, box[0]))
        i = np.argmax(box[:, 1])
        B = box[i]
        A = exp_box[i]
        C = exp_box[i + 2]
        AB_ = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        BC_ = math.sqrt((C[0] - B[0]) ** 2 + (C[1] - B[1]) ** 2)
        m = np.array([(A, AB_), (C, BC_)], dtype=object)
        j = np.argmax(m[:, 1])
        O = m[j, 0]
        # FIX: Éviter division par zéro
        if B[0] == O[0]:
            alfa = math.pi / 2
        else:
            alfa = math.atan((O[1] - B[1]) / (O[0] - B[0]))
        if alfa == 0:
            return alfa / math.pi * 180
        elif B[0] < O[0]:
            return -alfa / math.pi * 180
        else:
            return (math.pi - alfa) / math.pi * 180
        
    def adjust_angle(alfa, i=5):
        if i == 0:
            return alfa
        if -i < alfa < 90 - i:
            return -round(alfa / i) * i
        elif 90 - i < alfa < 90 + i:
            return round(alfa / i) * i - 180
        elif 90 + i < alfa < 180 + i:
            return 180 - round(alfa / i) * i
        else:
            return alfa

    def subimage(image, center, theta, width, height):
        if image is None or image.size == 0:
            return None
        if width <= 0 or height <= 0:
            return None
        
        padded_image = cv2.copyMakeBorder(image, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        shape = (padded_image.shape[1], padded_image.shape[0])
        padded_center = (center[0] + 300, center[1] + 300)
        matrix = cv2.getRotationMatrix2D(center=padded_center, angle=theta, scale=1)
        image = cv2.warpAffine(src=padded_image, M=matrix, dsize=shape)
        x, y = (int(padded_center[0] - width/2), int(padded_center[1] - height/2))
        x2, y2 = x + width, y + height

        if x < 0: x = 0
        if x2 > shape[0]: x2 = shape[0]
        if y < 0: y = 0
        if y2 > shape[1]: y2 = shape[1]

        if x >= x2 or y >= y2:
            return None
        
        image = image[y:y2, x:x2]
        return image

    def clean_h_lines(img_croped):
        if img_croped is None or img_croped.size == 0:
            return img_croped, None
        
        gray = cv2.cvtColor(img_croped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        h_kernel_size = int(img_croped.shape[1] * 0.8)
        if h_kernel_size > 0:
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
            detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                img_croped = cv2.drawContours(img_croped, [c], -1, (255, 255, 255), 3)
        
        v_kernel_size = int(img_croped.shape[1] * 0.9)
        if v_kernel_size > 0:
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
            detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                img_croped = cv2.drawContours(img_croped, [c], -1, (255, 255, 255), 3)
        
        return img_croped, thresh
    
    try:
        box = np.array(box, dtype=np.float32)
        rect = cv2.minAreaRect(box)
    except Exception:
        return None, [], 0
    
    # FIX: Éviter dimensions nulles
    rect_max = max(rect[1]) if max(rect[1]) > 0 else 1
    rect_min = min(rect[1]) if min(rect[1]) > 0 else 1
    
    angle = get_box_angle(box)
    angle = adjust_angle(angle, angle_t)
    w = int(w_multiplier * rect_max) + 1
    h = int(h_multiplier * rect_min) + 1
    
    if w <= 0 or h <= 0:
        return None, [], angle
    
    img_croped = subimage(img, rect[0], angle, w, h)
    
    if img_croped is None or img_croped.size == 0:
        return None, [], angle
    
    if w > 50 and h > 30:
        img_croped, thresh = clean_h_lines(img_croped)
    
    if img_croped is None or img_croped.size == 0:
        return None, [], angle
    
    gray = cv2.cvtColor(img_croped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    return img_croped, cnts, angle


def ocr_dimensions(img, detector, recognizer, alphabet_dim, frame, dim_boxes=[], cluster_thres=20, language='eng', max_img_size=2048, backg_save=False):
    """Pipeline principal pour l'OCR des dimensions"""
    
    if img is None or img.size == 0:
        return [], [], img, []
    
    # OCR dim_boxes first
    dimensions_ = []
    for d in dim_boxes:
        try:
            x, y = d.x - frame.x, d.y - frame.y
            if x < 0 or y < 0:
                continue
            if x + d.w > img.shape[1] or y + d.h > img.shape[0]:
                continue
            if d.w <= 4 or d.h <= 4:
                continue
            
            roi = img[y+2:y + d.h-4, x+2:x + d.w-4]
            if roi.size == 0:
                continue
            
            if d.h > d.w:
                roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            p = recognizer.recognize(image=roi)
            if any(char.isdigit() for char in p) and len(p) > 1:
                box = np.array([[x, y], [x + d.w, y], [x + d.w, y + d.h], [x, y + d.h]])
                dimensions_.append((p, box))
                img[y:y + d.h, x:x + d.w] = 255
        except Exception:
            continue

    # OCR the rest of the dimensions
    pipeline = Pipeline(
        recognizer=recognizer, 
        detector=detector, 
        alphabet_dimensions=alphabet_dim, 
        cluster_t=cluster_thres, 
        max_size=max_img_size, 
        language=language
    )
    
    try:
        dimensions, other_info, dim_pyt = pipeline.ocr_img_patches(img, 0.05)
    except ZeroDivisionError as e:
        print(f"[OCR] ZeroDivisionError in ocr_img_patches: {e}")
        dimensions, other_info, dim_pyt = [], [], []
    except Exception as e:
        print(f"[OCR] Error in ocr_img_patches: {e}")
        dimensions, other_info, dim_pyt = [], [], []
    
    dimensions.extend(dimensions_)
    
    # Mask detected regions
    for dim in dimensions:
        try:
            box = dim[1]
            pts = np.array([(box[0]), (box[1]), (box[2]), (box[3])]).astype(np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255))
        except Exception:
            continue
    
    for dim in other_info:
        try:
            box = dim[1]
            pts = np.array([(box[0]), (box[1]), (box[2]), (box[3])]).astype(np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255))
        except Exception:
            continue
    
    # Save background for synthetic data training
    if backg_save:
        try:
            backg_path = os.path.join(os.getcwd(), 'edocr2/tools/backgrounds')
            os.makedirs(backg_path, exist_ok=True)
            i = 0
            for root_dir, cur_dir, files in os.walk(backg_path):
                i += len(files)
            image_filename = os.path.join(backg_path, f'backg_{i + 1}.png')
            cv2.imwrite(image_filename, img)
        except Exception:
            pass
        
    return dimensions, other_info, img, dim_pyt