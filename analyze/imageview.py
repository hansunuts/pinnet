import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)

from dataclasses import dataclass
from dataclasses import field
import numpy as np
import open3d as o3d
from PIL import Image as im
from PIL import ImageDraw

from analyze.baseview import ConfigBaseView, BaseView

@dataclass
class ConfigImageView(ConfigBaseView):
    name:str='imageview'
    height:int=0
    width:int=0
    kp_size:int=2
    
class ImageView(BaseView):   
    def __init__(self, cfg:ConfigImageView=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigImageView()
        self.cfg = cfg
        self.draw_count = 0
    
    def setup(self, vis):
        super().setup(vis)
        self.images = {}
        self.img = o3d.geometry.Image()
        self.vis.add_geometry(self.img)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.img)

    def set(self, image:np.ndarray, zorder:int=0, alpha:float=1.0):
        '''
        Use the first set image size as the canvas size.
        Args:
            image: (h_img, w_img, dim_img) np, RGB [0, 255]
            zorder: higher is closer to the camera
            alpha: [0, 1]
        '''
        self.dirty = True
        self.vis.remove_geometry(self.img)
        
        (h_img, w_img, dim_img) = image.shape
        if self.cfg.height == 0:
            self.cfg.height = h_img
        if self.cfg.width == 0:
            self.cfg.width = w_img
        
        if dim_img == 3:
            image = np.pad(image, (0, 1), 'constant', constant_values=(0, int(alpha*255)))
        elif dim_img == 1:
            image = np.pad(image, (0, 2), 'maximum')
            image = np.pad(image, (0, 1), 'constant', constant_values=(0, int(alpha*255)))
        
        image = im.fromarray(image.astype(np.uint8), "RGBA")
        self.images[zorder] = image
        
        self.img = im.new(mode="RGBA", size=(self.cfg.width, self.cfg.height))
        
        for i in sorted(self.images.keys()):
            self.img.paste(self.images[i].resize((self.cfg.width, self.cfg.height)), (0, 0), self.images[i].resize((self.cfg.width, self.cfg.height)))
            
        self.img = self.img.convert("RGB")
        
        self.img = o3d.geometry.Image(np.array(self.img))
        self.vis.add_geometry(self.img)
        
    def set_kp(self, 
               keypoints:np.ndarray,  # (n, 2)
               color=(255, 0, 0, 255)
               ):
        self.dirty = True
        self.vis.remove_geometry(self.img)
        
        kp_image = im.new('RGBA', (self.cfg.width, self.cfg.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(kp_image)
        for kp in keypoints:
            draw.ellipse((kp[0]-self.cfg.kp_size, 
                          kp[1]-self.cfg.kp_size, 
                          kp[0]+self.cfg.kp_size, 
                          kp[1]+self.cfg.kp_size),
                         outline=color,
                         width=1)
        
        kp_image_np = np.array(kp_image, dtype=np.uint32)
        image = im.fromarray(kp_image_np.astype(np.uint8), "RGBA")
        self.images[1000+self.draw_count] = image
        self.img = im.new(mode="RGBA", size=(self.cfg.width, self.cfg.height))
        
        for i in sorted(self.images.keys()):
            self.img.paste(self.images[i].resize((self.cfg.width, self.cfg.height)), (0, 0), self.images[i].resize((self.cfg.width, self.cfg.height)))
            
        self.img = self.img.convert("RGB")
        
        self.img = o3d.geometry.Image(np.array(self.img))
        self.vis.add_geometry(self.img)

        self.draw_count += 1
        
    def set_line(self, 
                start:np.ndarray,      # (2) x, y
                end:np.ndarray,        # (2) x, y
                color=(255, 255, 255, 255),
                width:int=1            
                ):
        ''' draw a line on image
        Args:
            start: xy, start coordinate
            end: xy, end coordinate
            color: 
            width:
        '''
        self.dirty = True
        self.vis.remove_geometry(self.img)
        
        line_image = im.new('RGBA', (self.cfg.width, self.cfg.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(line_image)
        draw.line([(int(start[0]), int(start[1])), (int(end[0]), int(end[1]))], fill=color, width=width)
        
        line_image_np = np.array(line_image, dtype=np.uint32)
        image = im.fromarray(line_image_np.astype(np.uint8), "RGBA")
        self.images[1000+self.draw_count] = image
        self.img = im.new(mode="RGBA", size=(self.cfg.width, self.cfg.height))
        
        for i in sorted(self.images.keys()):
            self.img.paste(self.images[i].resize((self.cfg.width, self.cfg.height)), (0, 0), self.images[i].resize((self.cfg.width, self.cfg.height)))
            
        self.img = self.img.convert("RGB")
        
        self.img = o3d.geometry.Image(np.array(self.img))
        self.vis.add_geometry(self.img)

        self.draw_count += 1
        
    def get_current_image(self) -> np.ndarray:
        ''' get the current render image
        Returns:
            image_np    (h, w, 3)
        '''
        return np.asarray(self.img)
        
        
