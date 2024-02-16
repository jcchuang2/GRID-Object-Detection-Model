import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help = "path to image", type=str)
args = parser.parse_args()

CLASSES = ['N/A', 'pole']
NUM_CLASSES = 2
IMG_SIZE = 640
THRESHOLD = 0.9

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_trained_model():
    model = torch.hub.load('facebookresearch/detr',
                        'detr_resnet50',
                        pretrained=False,
                        num_classes=NUM_CLASSES)

    checkpoint = torch.load('outputs/checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                        strict=False)

    model.eval()

    return model


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(outputs, size, threshold=THRESHOLD):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)
  
    return probas_to_keep, bboxes_scaled


def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    data_path = "data/images/model_results/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plt.savefig(data_path + '/' + args.image_path.split('/')[-1])
    plt.show()


def run_worflow(my_image, my_model):
    
    img = transform(my_image).unsqueeze(0)

    outputs = my_model(img)

    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, my_image.size)
    plot_finetuned_results(my_image,probas_to_keep, bboxes_scaled)


if __name__ == "__main__":
    model = load_trained_model()
    im = Image.open(args.image_path).convert('RGB')

    run_worflow(im, model)