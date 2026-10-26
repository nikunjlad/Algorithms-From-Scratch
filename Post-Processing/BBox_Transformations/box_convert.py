


def xyxy_to_xywh(b):
    x1,y1,x2,y2 = b
    return [x1,y1,x2-x1,y2-y1]

def xywh_to_xyxy(b):
    x,y,w,h = b
    return [x,y,x+w,y+h]

def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    return [(x2-x1) / 2, (y2-y1) / 2, x2-x1, y2-y1]

def cxcywh_to_xyxy(b):
    cx,cy,w,h = b
    return [cx - (w / 2), cy - (h / 2), cx + (w / 2), cy + (h / 2)]

def normalize(b,W,H):
    x1,y1,x2,y2 = b
    return [x1/W,y1/H,x2/W,y2/H]

def denormalize(b,W,H):
    x1,y1,x2,y2 = b
    return [x1*W,y1*H,x2*W,y2*H]

def letterbox_params(orig_w,orig_h,target_w,target_h):
    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
    pad_left = (target_w - resized_w) / 2
    pad_top = (target_h - resized_h) / 2
    return scale, pad_left, pad_top

def box_to_letterbox(b,scale,pad_left,pad_top):
    x1,y1,x2,y2 = b
    return [x1*scale + pad_left, y1*scale + pad_top, x2*scale + pad_left, y2*scale + pad_top]

def letterbox_to_box(b,scale,pad_left,pad_top):
    x1,y1,x2,y2 = b
    return [(x1 - pad_left) / scale, (y1 - pad_top) / scale, (x2 - pad_left) / scale, (y2 - pad_top) / scale]