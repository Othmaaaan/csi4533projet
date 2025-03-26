import json
import numpy as np
from collections import defaultdict

# Seuil d'association basé sur l'IoU
IOU_THRESHOLD = 0.5
# Nombre maximum de frames sans association avant de terminer un tracklet
MAX_MISSING = 20
# Seuil minimal de détections dans un tracklet pour le conserver
MIN_TRACKLET_LENGTH = 50

# Fonction pour calculer l'IoU entre deux boîtes [x, y, w, h]
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    x_inter = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_inter = max(0, min(y1_max, y2_max) - max(y1, y2))
    intersection = x_inter * y_inter
    union = (w1 * h1) + (w2 * h2) - intersection
    return intersection / union if union != 0 else 0

# Fonction pour prédire la prochaine position d'un tracklet
def predict_bbox(tracklet):
    # Si le tracklet n'a qu'une seule détection, la prédiction est identique à la dernière position
    if len(tracklet["detections"]) == 1:
        return tracklet["detections"][-1]["bbox"]
    # Sinon, calculer le déplacement (dx, dy) entre les deux dernières positions
    last_bbox = tracklet["detections"][-1]["bbox"]
    second_last_bbox = tracklet["detections"][-2]["bbox"]
    
    # Calculer les centres
    last_center = (last_bbox[0] + last_bbox[2] / 2, last_bbox[1] + last_bbox[3] / 2)
    second_last_center = (second_last_bbox[0] + second_last_bbox[2] / 2, second_last_bbox[1] + second_last_bbox[3] / 2)
    dx = last_center[0] - second_last_center[0]
    dy = last_center[1] - second_last_center[1]
    # La prédiction est la dernière boîte décalée de (dx, dy)
    predicted_bbox = [last_bbox[0] + dx, last_bbox[1] + dy, last_bbox[2], last_bbox[3]]
    return predicted_bbox

# Charger le fichier d'annotations
annotations_file = '/Users/macbookair/Downloads/2021-11-20_lunch_2_cam0.json'
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Construire un dictionnaire d'annotations par image pour faciliter l'accès
frame_detections = defaultdict(list)
for detection in annotations['annotations']:
    # Filtrer sur la confiance
    if detection['confidence'] >= 0.5:
        frame_detections[detection['image_id']].append(detection)

# Obtenir la liste des images triées par image_id
frames = sorted(annotations['images'], key=lambda x: x['id'])

# Listes pour tracklets actifs et terminés
active_tracklets = []
finished_tracklets = []

# Générateur d'identifiants de tracklet
next_tracklet_id = 1

# Parcourir les frames dans l'ordre
for frame in frames:
    image_id = frame['id']
    current_detections = frame_detections.get(image_id, [])
    # Pour marquer les détections déjà associées
    assigned = [False] * len(current_detections)
    
    # Premièrement, tenter d'associer chaque tracklet actif aux détections de la frame courante
    for tracklet in active_tracklets:
        predicted_box = predict_bbox(tracklet)
        best_iou = 0
        best_det_idx = -1
        # Chercher la détection avec le meilleur IoU par rapport à la boîte prédite
        for i, det in enumerate(current_detections):
            if assigned[i]:
                continue
            iou = compute_iou(predicted_box, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det_idx = i
        # Si le meilleur IoU dépasse le seuil, associer la détection au tracklet
        if best_iou >= IOU_THRESHOLD and best_det_idx != -1:
            det = current_detections[best_det_idx]
            tracklet["detections"].append({
                "frame": image_id,
                "bbox": det['bbox'],
                "object_id": det['id']
            })
            tracklet["last_frame"] = image_id
            tracklet["missing"] = 0
            assigned[best_det_idx] = True
        else:
            # Sinon, incrémenter le compteur de frames manquantes pour ce tracklet
            tracklet["missing"] += 1

    # Terminer les tracklets inactifs depuis trop longtemps
    still_active = []
    for tracklet in active_tracklets:
        if tracklet["missing"] > MAX_MISSING:
            finished_tracklets.append(tracklet)
        else:
            still_active.append(tracklet)
    active_tracklets = still_active

    # Pour chaque détection non associée dans la frame, démarrer un nouveau tracklet
    for i, det in enumerate(current_detections):
        if not assigned[i]:
            new_tracklet = {
                "tracklet_id": next_tracklet_id,
                "detections": [{
                    "frame": image_id,
                    "bbox": det['bbox'],
                    "object_id": det['id']
                }],
                "last_frame": image_id,
                "missing": 0
            }
            active_tracklets.append(new_tracklet)
            next_tracklet_id += 1

# À la fin, ajouter les tracklets encore actifs aux tracklets terminés
finished_tracklets.extend(active_tracklets)

# Filtrer pour ne conserver que les tracklets contenant au moins MIN_TRACKLET_LENGTH détections
valid_tracklets = []
for tracklet in finished_tracklets:
    if len(tracklet["detections"]) >= MIN_TRACKLET_LENGTH:
        # Pour la sortie, on utilise l'image_id et object_id de la première détection
        first_det = tracklet["detections"][0]
        valid_tracklets.append({
            "object_id": first_det["object_id"],
            "tracklet_id": tracklet["tracklet_id"],
            "image_id": first_det["frame"]
        })

# Sauvegarder le résultat dans un fichier JSON
output = {"tracklets": valid_tracklets}
output_file = '/Users/macbookair/Downloads/output.json'
with open(output_file, 'w') as f:
    json.dump(output, f)

print(f"Le suivi est terminé ! {len(valid_tracklets)} tracklets valides ont été sauvegardés dans {output_file}")
