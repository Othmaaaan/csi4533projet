import json
import os
import cv2
import numpy as np
from collections import defaultdict

# Seuil d'association basé sur l'IoU
IOU_THRESHOLD = 0.5
# Nombre maximum de frames sans association avant de terminer un tracklet
MAX_MISSING = 20
# Seuil minimal de détections dans un tracklet pour le conserver
MIN_TRACKLET_LENGTH = 50

# Dossier où l'on va sauvegarder les images annotées
OUTPUT_IMAGES_DIR = "output_images"
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def compute_iou(box1, box2):
    """
    Calcule l'IoU (Intersection over Union) entre deux boîtes [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    x_inter = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_inter = max(0, min(y1_max, y2_max) - max(y1, y2))
    intersection = x_inter * y_inter
    union = (w1 * h1) + (w2 * h2) - intersection
    return intersection / union if union != 0 else 0

def predict_bbox(tracklet):
    """
    Prédit la prochaine position (bbox) d'un tracklet à partir de ses deux dernières détections.
    """
    if len(tracklet["detections"]) == 1:
        # Si une seule détection, on réutilise la même bbox
        return tracklet["detections"][-1]["bbox"]
    
    last_bbox = tracklet["detections"][-1]["bbox"]
    second_last_bbox = tracklet["detections"][-2]["bbox"]
    
    # Calculer les centres
    last_center = (last_bbox[0] + last_bbox[2] / 2, last_bbox[1] + last_bbox[3] / 2)
    second_last_center = (second_last_bbox[0] + second_last_bbox[2] / 2,
                          second_last_bbox[1] + second_last_bbox[3] / 2)
    dx = last_center[0] - second_last_center[0]
    dy = last_center[1] - second_last_center[1]
    
    # Décaler la dernière bbox de (dx, dy)
    predicted_bbox = [
        last_bbox[0] + dx,
        last_bbox[1] + dy,
        last_bbox[2],
        last_bbox[3]
    ]
    return predicted_bbox

# Charger le fichier d'annotations
annotations_file = '2021-11-20_lunch_2_cam0.json'
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
    
    print(f"\n=== Traitement de l'image ID: {image_id} ===")
    print(f"Nombre de détections dans cette image : {len(current_detections)}")
    
    # Pour marquer les détections déjà associées
    assigned = [False] * len(current_detections)
    
    # Premièrement, associer chaque tracklet actif aux détections de la frame courante
    print(f"Tracklets actifs avant association : {len(active_tracklets)}")
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
            
            print(f"   → Tracklet {tracklet['tracklet_id']} associé à la détection {det['id']} (IoU={best_iou:.2f})")
        else:
            # Sinon, incrémenter le compteur de frames manquantes pour ce tracklet
            tracklet["missing"] += 1
            if best_iou > 0:
                print(f"   → Tracklet {tracklet['tracklet_id']} NON associé (IoU max={best_iou:.2f} < seuil={IOU_THRESHOLD})")
            else:
                print(f"   → Tracklet {tracklet['tracklet_id']} NON associé (pas de détection compatible)")
    
    # Terminer les tracklets inactifs depuis trop longtemps
    still_active = []
    for tracklet in active_tracklets:
        if tracklet["missing"] > MAX_MISSING:
            finished_tracklets.append(tracklet)
            print(f"      X Tracklet {tracklet['tracklet_id']} terminé (missing > {MAX_MISSING})")
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
            print(f"   → Nouveau tracklet créé : ID={next_tracklet_id} pour la détection {det['id']}")
            next_tracklet_id += 1
    
    print(f"Tracklets actifs après association : {len(active_tracklets)}")
    
    #--------------------------------------------------------------------------
    # Génération et enregistrement de l'image avec les boîtes englobantes
    #--------------------------------------------------------------------------
    
    file_name_fix = frame["file_name"].split("/")[-1]
    image_path = os.path.join("images", file_name_fix)
    img = cv2.imread(image_path)
    
    if img is None:
        # En cas de problème de lecture (chemin invalide, etc.)
        print(f"Impossible de lire l'image : {image_path}. On passe à la suivante.")
        continue
    
    # Dessiner la bounding box pour chaque tracklet qui a une détection sur cette frame
    for tracklet in active_tracklets:
        # Chercher la dernière détection du tracklet
        if len(tracklet["detections"]) == 0:
            continue
        
        last_det = tracklet["detections"][-1]
        # Vérifier si la dernière détection correspond à cette frame
        if last_det["frame"] == image_id:
            (x, y, w, h) = last_det["bbox"]
            
            # Dessiner un rectangle (bbox)
            cv2.rectangle(
                img,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),  # couleur vert (BGR)
                2             # épaisseur
            )
            # Écrire l'ID du tracklet au-dessus de la bbox
            cv2.putText(
                img,
                f"ID: {tracklet['tracklet_id']}",
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,          # taille du texte
                (0, 255, 0),  # couleur du texte (vert)
                2             # épaisseur du trait
            )
    
    # Enregistrer l'image annotée dans le dossier output_images
    output_path = os.path.join(OUTPUT_IMAGES_DIR, f"frame_{image_id}.jpg")
    cv2.imwrite(output_path, img)
    print(f"Image annotée sauvegardée : {output_path}")
    #--------------------------------------------------------------------------

# À la fin, ajouter les tracklets encore actifs aux tracklets terminés
finished_tracklets.extend(active_tracklets)

# Filtrer pour ne conserver que les tracklets contenant au moins MIN_TRACKLET_LENGTH détections
valid_tracklets = []
for tracklet in finished_tracklets:
    if len(tracklet["detections"]) >= MIN_TRACKLET_LENGTH:
        # On retient, par exemple, object_id et image_id de la première détection
        first_det = tracklet["detections"][0]
        valid_tracklets.append({
            "object_id": first_det["object_id"],
            "tracklet_id": tracklet["tracklet_id"],
            "image_id": first_det["frame"]
        })

# Sauvegarder le résultat final dans un fichier JSON
output = {"tracklets": valid_tracklets}
output_file = 'output.json'
with open(output_file, 'w') as f:
    json.dump(output, f)

print(f"\nLe suivi est terminé ! {len(valid_tracklets)} tracklets valides ont été sauvegardés dans {output_file}")
print(f"Les images annotées sont enregistrées dans le dossier: {OUTPUT_IMAGES_DIR}")
