from vis_utils import cam_renderer
from glob import glob

classifynet_renderer = cam_renderer('checkpoint/resnetlight_classify_epoch320.model', ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
for pic_name in glob("images/inputs/*"):
    classifynet_renderer.render(pic_name, 'images/classifynet_results/')


rotatenet_renderer = cam_renderer('checkpoint/resnetlight_rotate_epoch100.model', ['up', 'left', 'down', 'right'], ifrotate = True)
for pic_name in glob("images/inputs/*"):
    rotatenet_renderer.render(pic_name, 'images/rotatenet_results/')
