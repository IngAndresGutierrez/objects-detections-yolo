# Prompts del VLM

Registra aquí el **modelo exacto y la web** usados (p. ej. "Gemini 2.5 Flash — aistudio.google.com",
"Qwen3-VL-Plus — chat.qwen.ai"), la fecha y la temperatura, si la interfaz permite fijarla (usa 0).

- Modelo: 
- Web / herramienta: 
- Fecha: 
- Temperatura / otros ajustes: 

**Una conversación nueva por imagen** (para que no arrastre el contexto de imágenes anteriores).
Pega la respuesta **completa y sin editar** en `responses/<split>/<prompt>/<id_imagen>.txt`.

## Fase 1 · Validación (10 imágenes de `dataset/images/val/`): P1 y P2

### P1_simple

```
Detect every car in the image. Return only a JSON list, one entry per car, in the form
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "car"}] with coordinates normalized to 0-1000.
Return [] if there are no cars.
```

### P2_voc

```
Detect every car in the image. A "car" includes passenger cars, vans, minivans, pickups, SUVs,
jeeps and race cars; it does NOT include buses, lorries, motorbikes or bicycles.
Include cars that are partially visible, occluded, far away or cut by the image border, and draw
each box around the full visible extent of the car. Do not return duplicated boxes.
Return only a JSON list, one entry per car, in the form
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "car"}] with coordinates normalized to 0-1000.
Return [] if there are no cars.
```

## Fase 2 · Test (5 imágenes de `dataset/images/test/`)

Solo cuando se haya elegido el prompt con validación: el prompt ganador (se crea su carpeta en
`responses/test/`) y además P3.

### P3_descriptivo (pregunta compleja, análisis cualitativo)

```
Describe every car in the image. For each one give: colour, type (sedan, convertible, van, race car...),
orientation relative to the camera, whether it is occluded or truncated, and its bounding box.
Return only a JSON list in the form
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "car", "colour": "...", "type": "...",
  "orientation": "...", "occluded": true/false}]
with coordinates normalized to 0-1000.
```

> Si usas Qwen: Qwen3-VL suele responder con `"bbox_2d": [x1, y1, x2, y2]` en 0-1000 y Qwen2.5-VL en
> píxeles absolutos. No pasa nada: pega la respuesta tal cual e indica arriba el modelo; el conversor
> lo tiene en cuenta.
