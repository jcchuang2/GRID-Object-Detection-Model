# Modifies the DETR custom training script in order to match our data naming convention
with open('detr/datasets/custom.py', 'r+') as file:
    content = file.read()
    
    modified_content = content.replace("train2017", "image_data/train")
    modified_content = modified_content.replace("val2017", "image_data/val")
    
    modified_content = modified_content.replace("custom_train.json", "image_annotations.json")
    modified_content = modified_content.replace("custom_val.json", "image_annotations.json")
    
    file.seek(0)
    file.write(modified_content)
    file.truncate()
    
    file.close()