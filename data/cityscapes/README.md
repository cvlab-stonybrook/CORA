# CORA instruction sets for Cityscapes

`instructions/train/` holds the instructions for the 815 training images that are labeled in at least one of the
partitions 1/30, 1/16, 1/8 and 1/4; `instructions/val/` those for the 500 validation images (the two LLM-written sets
cover 100 of them). One JSON object per line.

| File | Train | Val | Content |
|---|---|---|---|
| `semantic.jsonl` | 9,461 | 6,005 | "Please segment the road in this image." (templates, whole class) |
| `attribute.jsonl` | 14,570 | 1,793 | LLM-written description of one object |
| `conditional.jsonl` | 14,643 | 1,793 | LLM-written, the target relative to an anchor object `<REF>` |
| `conditional_templated.jsonl` | 3,260 | 2,000 | "Segment the cars on the left of `<REF>`." (rule-based) |

Fields of a record:

* `id`, `image_id`, `split`, `set`: `image_id` is the Cityscapes name, for example `train/jena/jena_000078_000019`
* `query`: the instruction; `<REF>` marks where the anchor object is referred to
* `target`: `{"label_id", "label_name", "region_ids"}`, the object to segment (`label_id` is the Cityscapes train id)
* `anchor`: the reference object of a conditional instruction, in the same form (`null` otherwise)
* `relation`, `description`, `meta`: the spatial relation, the description used, and how the record was made

## Masks

The masks are not part of this repository: they are the Cityscapes annotations, which the Cityscapes licence does not
allow to be redistributed. Build them from your own copy of Cityscapes (see the main README):

```bash
for p in 1_30 1_16 1_8 1_4; do python tools/build_objects.py --partition $p --which labeled --workers 8; done
python tools/build_objects.py --split val --workers 8
```

This writes, for every image, `objects/<split>/<image_id>.png` (a uint16 region map: 0 = no region, otherwise the
index of a region) and `objects/<split>/<image_id>.json` (the table of its regions: `index`, `region_id`, `label_id`,
`label_name`, `area`, `bbox`, `centroid`). A region is one connected component (8-connectivity) of a class in the
ground truth, numbered per class in raster order, so the `region_ids` of the instructions always refer to the same
regions. The mask of a target or anchor is the union of its `region_ids`; an empty list means the whole class.

`masks.py` (numpy and Pillow) returns these masks and can write them as images:

```bash
python masks.py show instructions/val/conditional.jsonl 0            # one record and the size of its masks
python masks.py export instructions/val/conditional.jsonl out_dir    # <line>_target.png / <line>_anchor.png (0 / 255)
```
