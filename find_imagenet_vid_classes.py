"""Find all unique class IDs in ImageNet-VID annotations."""
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

def find_classes_in_split(ann_root: str):
    classes = set()
    count = 0
    
    for root, dirs, files in os.walk(ann_root):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                try:
                    ann = ET.parse(xml_path)
                    for obj in ann.findall('object'):
                        name = obj.find('name').text
                        classes.add(name)
                        count += 1
                except:
                    pass
    
    return sorted(classes), count

# Check val split
val_ann = r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Annotations\VID\val"
classes, obj_count = find_classes_in_split(val_ann)
print(f"Found {len(classes)} unique classes in val split ({obj_count} objects)")
print("\nClasses:")
for cls in classes:
    print(f"  {cls}")
