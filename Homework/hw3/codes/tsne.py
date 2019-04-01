from vis_utils import render_tsne

render_tsne('checkpoint/resnetlight_classify_epoch320.model', ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
render_tsne('checkpoint/resnetlight_rotate_epoch100.model', ['up', 'left', 'down', 'right'])