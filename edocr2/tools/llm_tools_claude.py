"""
LLM Tools pour correction OCR avec Claude Anthropic
Compatible avec le workflow edocr2
"""

import os
import io
import ast
import re
import json
import base64
from PIL import Image


def convert_img_to_base64(img):
    """Convertit une image numpy/PIL en base64 pour l'API"""
    if hasattr(img, 'shape'):  # numpy array
        pil_img = Image.fromarray(img)
    else:
        pil_img = img
    
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode('utf-8')


def claude_correct_ocr(img, ocr_results: list, api_key: str = None) -> dict:
    """
    Utilise Claude pour corriger et améliorer les résultats OCR d'un plan technique.
    
    Fonctionnalités:
    - Corrige les erreurs de lecture OCR (0/O, 1/l, etc.)
    - Fusionne les tolérances empilées (44.60/44.45 → 44.525 ±0.075)  
    - Détecte les cotes manquées par l'OCR classique
    - Améliore la classification des types (diameter, radius, angle, etc.)
    
    Args:
        img: Image numpy array ou PIL
        ocr_results: Liste des résultats OCR bruts [{'id':1, 'value':'44.60', 'x':100, 'y':200}, ...]
        api_key: Clé API Anthropic (optionnel, utilise ANTHROPIC_API_KEY sinon)
    
    Returns:
        dict: {
            'corrected': [...],  # Liste corrigée
            'corrections_made': [...],  # Liste des corrections appliquées
            'missed_dimensions': [...]  # Cotes trouvées par le LLM mais pas par l'OCR
        }
    """
    try:
        import anthropic
    except ImportError:
        print("[LLM] Module anthropic non installé, pip install anthropic")
        return {'corrected': ocr_results, 'corrections_made': [], 'missed_dimensions': []}
    
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("[LLM] Pas de clé API Anthropic disponible")
        return {'corrected': ocr_results, 'corrections_made': [], 'missed_dimensions': []}
    
    client = anthropic.Anthropic(api_key=api_key)
    img_base64 = convert_img_to_base64(img)
    
    # Préparer le résumé OCR
    ocr_summary = []
    for item in ocr_results:
        ocr_summary.append({
            'id': item.get('id'),
            'value': item.get('value', ''),
            'tolerance': item.get('tolerance', ''),
            'type': item.get('type', 'dimension'),
            'x': item.get('x', 0),
            'y': item.get('y', 0)
        })
    
    prompt = f"""Tu es un expert en lecture de plans industriels mécaniques. Tu dois analyser cette image et corriger/améliorer les résultats OCR.

RÉSULTATS OCR DÉTECTÉS:
{json.dumps(ocr_summary, indent=2, ensure_ascii=False)}

TÂCHES À EFFECTUER:

1. **FUSION DES TOLÉRANCES EMPILÉES** (PRIORITAIRE)
   Sur les plans techniques, les tolérances bilatérales sont affichées verticalement:
   - Valeur MAX en haut (ex: 44.60)
   - Valeur MIN en bas (ex: 44.45)
   
   Règles de détection:
   - Deux valeurs numériques proches verticalement (même X ±50px, Y différent de 15-50px)
   - Les valeurs doivent être proches (ratio < 1.15)
   - Même préfixe (⌀, Ø, R) ou pas de préfixe
   
   Calcul:
   - Valeur nominale = (max + min) / 2
   - Tolérance = ±(max - min) / 2
   
   Exemple: ⌀44.60 (y=100) + 44.45 (y=130) → ⌀44.525 ±0.075

2. **CORRECTION DES ERREURS OCR**
   - 0 confondu avec O
   - 1 confondu avec l ou I
   - 5 confondu avec S
   - 8 confondu avec B
   - Point décimal manquant ou mal placé

3. **CLASSIFICATION DES TYPES**
   - diameter: préfixe ⌀, Ø, ø, phi, ou contexte cylindre
   - radius: préfixe R, r
   - angle: contient °
   - thread: format Mxx×pas (filetage métrique)
   - gdt: symboles géométriques (⊥, ∥, ○, ◇, etc.)
   - dimension: autres mesures linéaires

4. **COTES MANQUÉES**
   Regarde l'image et liste les cotes visibles qui ne sont PAS dans les résultats OCR.

RETOURNE UNIQUEMENT CE JSON (pas d'explication):
{{
  "corrected": [
    {{"id": 1, "value": "44.525", "tolerance": "±0.075", "type": "diameter", "x": 100, "y": 100, "merged_from": [1, 2]}},
    {{"id": 3, "value": "25.4", "tolerance": "", "type": "dimension", "x": 200, "y": 300}}
  ],
  "corrections_made": [
    "Fusionné ⌀44.60/44.45 → ⌀44.525 ±0.075 (IDs 1+2)",
    "Corrigé '1O.5' → '10.5'"
  ],
  "missed_dimensions": [
    {{"value": "M8×1.25", "type": "thread", "approximate_location": "haut gauche"}},
    {{"value": "R2.5", "type": "radius", "approximate_location": "coin arrondi droite"}}
  ]
}}

RÈGLES:
- Garde les IDs originaux quand possible
- Pour les fusions, utilise le plus petit ID et indique merged_from
- Ne change PAS les positions x,y sauf fusion (utilise position du haut)
- missed_dimensions: seulement si tu vois clairement des cotes non détectées"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        response_text = response.content[0].text
        print(f"[LLM] Réponse reçue: {len(response_text)} caractères")
        
        # Extraire le JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            
            corrected = result.get('corrected', ocr_results)
            corrections = result.get('corrections_made', [])
            missed = result.get('missed_dimensions', [])
            
            print(f"[LLM] {len(corrections)} corrections, {len(missed)} cotes manquées trouvées")
            for c in corrections:
                print(f"[LLM]   → {c}")
            
            return {
                'corrected': corrected,
                'corrections_made': corrections,
                'missed_dimensions': missed
            }
        else:
            print("[LLM] Pas de JSON valide dans la réponse")
            return {'corrected': ocr_results, 'corrections_made': [], 'missed_dimensions': []}
            
    except Exception as e:
        print(f"[LLM] Erreur API Claude: {e}")
        return {'corrected': ocr_results, 'corrections_made': [], 'missed_dimensions': []}


def claude_extract_dimensions(img, api_key: str = None) -> list:
    """
    Utilise Claude pour extraire TOUTES les dimensions d'un plan (sans OCR préalable).
    Équivalent à gpt4_dim() mais avec Claude.
    
    Args:
        img: Image numpy array ou PIL
        api_key: Clé API Anthropic
    
    Returns:
        list: Liste de dimensions extraites
    """
    try:
        import anthropic
    except ImportError:
        return []
    
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return []
    
    client = anthropic.Anthropic(api_key=api_key)
    img_base64 = convert_img_to_base64(img)
    
    prompt = """Tu es un système OCR spécialisé dans la lecture de plans mécaniques industriels.

Extrais TOUTES les cotes et dimensions visibles sur ce plan:
- Dimensions linéaires (longueur, largeur, hauteur)
- Diamètres (⌀, Ø)
- Rayons (R)
- Angles (°)
- Filetages (Mxx)
- Tolérances (±, +/-)
- Symboles GD&T

Pour chaque dimension, indique:
- La valeur exacte lue
- La tolérance si présente (format: ±X ou +X/-Y)
- Le type (dimension, diameter, radius, angle, thread, gdt)

RETOURNE UNIQUEMENT UNE LISTE PYTHON:
[
    {"value": "⌀42", "tolerance": "±0.1", "type": "diameter"},
    {"value": "25.5", "tolerance": "+0.05/-0.02", "type": "dimension"},
    {"value": "M8×1.25", "tolerance": "", "type": "thread"},
    {"value": "45°", "tolerance": "", "type": "angle"}
]"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png", 
                                "data": img_base64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        response_text = response.content[0].text
        
        # Extraire la liste
        list_match = re.search(r'\[[\s\S]*\]', response_text)
        if list_match:
            return json.loads(list_match.group())
        
        return []
        
    except Exception as e:
        print(f"[LLM] Erreur extraction Claude: {e}")
        return []


def merge_stacked_tolerances_local(bullage_items: list, 
                                    vertical_threshold: int = 45,
                                    horizontal_threshold: int = 60) -> list:
    """
    Version LOCALE (sans LLM) de fusion des tolérances empilées.
    Fallback si pas d'API key disponible.
    
    Détecte les paires de valeurs empilées:
    - ⌀44.60 (y=100)
    - 44.45  (y=130)  
    → Fusionne en ⌀44.525 ±0.075
    
    Args:
        bullage_items: Liste des items OCR
        vertical_threshold: Distance Y max entre deux valeurs empilées
        horizontal_threshold: Distance X max pour considérer comme alignées
    
    Returns:
        Liste avec tolérances fusionnées
    """
    if not bullage_items or len(bullage_items) < 2:
        return bullage_items
    
    # Séparer numériques et autres
    numeric_items = []
    other_items = []
    
    for item in bullage_items:
        value = str(item.get('value', ''))
        
        # Extraire préfixe
        prefix = ''
        if value.startswith(('⌀', 'Ø', 'ø')):
            prefix = '⌀'
            value_clean = value[1:].strip()
        elif value.upper().startswith('R') and len(value) > 1 and value[1].isdigit():
            prefix = 'R'
            value_clean = value[1:].strip()
        else:
            value_clean = value.strip()
        
        # Vérifier si numérique
        try:
            num_val = float(value_clean.replace(',', '.'))
            numeric_items.append({
                **item,
                '_prefix': prefix,
                '_num_value': num_val,
                '_clean_value': value_clean
            })
        except ValueError:
            other_items.append(item)
    
    if len(numeric_items) < 2:
        return bullage_items
    
    # Trier par X puis Y
    numeric_items.sort(key=lambda d: (d.get('x', 0), d.get('y', 0)))
    
    merged = []
    skip_ids = set()
    
    for i, item1 in enumerate(numeric_items):
        if item1['id'] in skip_ids:
            continue
        
        # Chercher une paire
        best_match = None
        best_distance = float('inf')
        
        for j, item2 in enumerate(numeric_items):
            if i == j or item2['id'] in skip_ids:
                continue
            
            # Même préfixe requis
            if item1['_prefix'] != item2['_prefix']:
                continue
            
            # Alignement horizontal
            dx = abs(item1.get('x', 0) - item2.get('x', 0))
            if dx > horizontal_threshold:
                continue
            
            # Distance verticale
            dy = abs(item1.get('y', 0) - item2.get('y', 0))
            if dy < 10 or dy > vertical_threshold:
                continue
            
            # Valeurs proches
            v1, v2 = item1['_num_value'], item2['_num_value']
            if v1 == 0 or v2 == 0:
                continue
            ratio = max(v1, v2) / min(v1, v2)
            if ratio > 1.15:
                continue
            
            if dy < best_distance:
                best_distance = dy
                best_match = item2
        
        if best_match:
            # Fusionner
            v1 = item1['_num_value']
            v2 = best_match['_num_value']
            
            max_val = max(v1, v2)
            min_val = min(v1, v2)
            nominal = (max_val + min_val) / 2
            tol = (max_val - min_val) / 2
            
            # Formater
            prefix = item1['_prefix']
            
            # Nombre de décimales = max des deux originaux
            dec1 = len(item1['_clean_value'].split('.')[-1]) if '.' in item1['_clean_value'] else 0
            dec2 = len(best_match['_clean_value'].split('.')[-1]) if '.' in best_match['_clean_value'] else 0
            decimals = max(dec1, dec2)
            
            if decimals == 0:
                nominal_str = f"{prefix}{int(nominal)}"
                tol_str = f"±{int(tol)}" if tol == int(tol) else f"±{tol:.2f}"
            else:
                nominal_str = f"{prefix}{nominal:.{decimals}f}"
                tol_str = f"±{tol:.{decimals}f}"
            
            # Position = celle du haut
            top_item = item1 if item1.get('y', 0) < best_match.get('y', 0) else best_match
            
            merged_item = {
                'id': min(item1['id'], best_match['id']),
                'type': 'diameter' if prefix == '⌀' else ('radius' if prefix == 'R' else item1.get('type', 'dimension')),
                'value': nominal_str,
                'tolerance': tol_str,
                'x': top_item.get('x', 0),
                'y': top_item.get('y', 0),
                'w': max(item1.get('w', 50), best_match.get('w', 50)),
                'h': abs(item1.get('y', 0) - best_match.get('y', 0)) + max(item1.get('h', 20), best_match.get('h', 20))
            }
            
            print(f"[MERGE] {item1.get('value')}/{best_match.get('value')} → {nominal_str} {tol_str}")
            
            merged.append(merged_item)
            skip_ids.add(item1['id'])
            skip_ids.add(best_match['id'])
        else:
            # Pas de fusion
            clean_item = {k: v for k, v in item1.items() if not k.startswith('_')}
            merged.append(clean_item)
    
    # Ajouter les non-numériques
    merged.extend(other_items)
    
    # Renuméroter
    merged.sort(key=lambda d: (d.get('y', 0), d.get('x', 0)))
    for i, item in enumerate(merged):
        item['id'] = i + 1
    
    return merged


def ask_claude(messages: list, api_key: str = None, max_tokens: int = 3000) -> str:
    """
    Fonction générique pour poser une question à Claude.
    Équivalent à ask_gpt() mais avec Claude.
    
    Args:
        messages: Liste de messages au format Claude
        api_key: Clé API
        max_tokens: Nombre max de tokens en réponse
    
    Returns:
        Réponse texte de Claude
    """
    try:
        import anthropic
    except ImportError:
        return ""
    
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return ""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        print(f"[LLM] Erreur ask_claude: {e}")
        return ""


# ============== EXEMPLE D'UTILISATION ==============

if __name__ == "__main__":
    import cv2
    
    # Test avec une image
    img = cv2.imread("test_drawing.png")
    
    # Résultats OCR simulés (comme sortis de edocr2)
    ocr_results = [
        {'id': 1, 'value': '⌀44.60', 'tolerance': '', 'type': 'diameter', 'x': 100, 'y': 100},
        {'id': 2, 'value': '44.45', 'tolerance': '', 'type': 'dimension', 'x': 105, 'y': 135},
        {'id': 3, 'value': '36.0', 'tolerance': '', 'type': 'dimension', 'x': 200, 'y': 150},
        {'id': 4, 'value': '35.5', 'tolerance': '', 'type': 'dimension', 'x': 205, 'y': 185},
        {'id': 5, 'value': 'M42×1.5', 'tolerance': '', 'type': 'thread', 'x': 300, 'y': 100},
    ]
    
    # Test fusion locale (sans API)
    print("=== Test fusion locale ===")
    merged = merge_stacked_tolerances_local(ocr_results)
    for item in merged:
        print(f"  {item['id']}: {item['value']} {item.get('tolerance', '')}")
    
    # Test avec Claude (si API key disponible)
    if os.environ.get('ANTHROPIC_API_KEY'):
        print("\n=== Test correction Claude ===")
        result = claude_correct_ocr(img, ocr_results)
        print(f"Corrections: {result['corrections_made']}")
        print(f"Cotes manquées: {result['missed_dimensions']}")